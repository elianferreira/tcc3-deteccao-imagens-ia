"""Campanha experimental (Etapa 4 da Secao 3.5.2).

Cada tecnica e avaliada nos tres protocolos definidos na Secao 3.6.1 --
padrao (in-distribution), generalizacao (OOD) e robustez -- e todas as metricas
sao registradas em arquivos CSV para analise posterior.

Prevencao de vazamento no ajuste de T05
---------------------------------------
T05 e ajustada sobre os escores produzidos por T01-T04 no subconjunto de
VALIDACAO, e nao no de treinamento. T01 e T03 sao ajustadas sobre o conjunto de
treinamento e, sobre ele, produzem escores otimistas que nao representam seu
comportamento em dados novos. Ajustar a fusao sobre esses escores levaria T05 a
subestimar sistematicamente o peso das tecnicas treinadas localmente. O
subconjunto de validacao nao participa do ajuste de parametros de T01 e T03 --
apenas da selecao de hiperparametros e do criterio de parada -- e portanto
fornece escores com distribuicao mais proxima da de teste.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

import numpy as np

from ..config import RESULTS_DIR, TECHNIQUE_NAMES, TRAINING_SEEDS
from ..data.manifest import as_arrays, ManifestEntry, read_manifest
from ..data.perturbations import Perturbation, iter_perturbations, materialize
from ..metrics import compute_metrics, metrics_per_generator
from ..seeds import set_deterministic
from ..techniques.base import BaseTechnique, TechniqueError
from ..techniques.t05_fusion import SOURCE_TECHNIQUES, T05Fusion, build_score_matrix


@dataclass
class ProtocolResult:
    protocol: str
    technique_id: str
    condition: str = "clean"
    metrics: dict = field(default_factory=dict)
    per_generator: dict = field(default_factory=dict)
    probabilities: np.ndarray | None = None
    error: str | None = None
    seed: int | None = None

    @property
    def failed(self) -> bool:
        return self.error is not None

    def to_row(self) -> dict:
        row = {
            "protocol": self.protocol,
            "technique": self.technique_id,
            "technique_name": TECHNIQUE_NAMES.get(self.technique_id, self.technique_id),
            "condition": self.condition,
            "seed": self.seed if self.seed is not None else "",
            "status": "erro" if self.failed else "ok",
            "error": self.error or "",
        }
        row.update({k: v for k, v in self.metrics.items()})
        return row


class ExperimentRunner:
    """Orquestra treinamento, inferencia e registro de metricas."""

    def __init__(
        self,
        manifest_path: Path,
        techniques: dict[str, BaseTechnique],
        results_dir: Path = RESULTS_DIR,
        cache_scores: bool = True,
    ) -> None:
        self.manifest_path = Path(manifest_path)
        self.techniques = techniques
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.cache_scores = cache_scores
        self.results: list[ProtocolResult] = []

    # ------------------------------------------------------------------
    # Utilidades
    # ------------------------------------------------------------------

    def _load_split(self, split: str) -> tuple[list[Path], np.ndarray, np.ndarray]:
        entries = read_manifest(self.manifest_path, split=split)
        if not entries:
            raise ValueError(f"particao '{split}' vazia no manifesto {self.manifest_path}")
        return as_arrays(entries)

    def _score_cache_path(self, protocol: str, technique_id: str, condition: str, split: str) -> Path:
        cache_dir = self.results_dir / "scores"
        cache_dir.mkdir(parents=True, exist_ok=True)
        return cache_dir / f"{protocol}__{technique_id}__{condition}__{split}.npy"

    def _infer(
        self,
        technique: BaseTechnique,
        paths: Sequence[Path],
        protocol: str,
        condition: str,
        split: str,
    ) -> tuple[np.ndarray, float]:
        """Inferencia com cache em disco.

        A inferencia de T02 e T04 e cara e deterministica; reaproveita-la evita
        repetir o custo entre protocolos que compartilham o mesmo conjunto.

        Retorna ``(probabilidades, ms_por_imagem)``. Quando os escores vem do
        cache o tempo e NaN: reportar o tempo de leitura do disco como tempo de
        inferencia falsearia a medicao exigida por RNF01.
        """
        cache_path = self._score_cache_path(protocol, technique.technique_id, condition, split)
        if self.cache_scores and cache_path.exists():
            cached = np.load(cache_path)
            if cached.shape == (len(paths),):
                return cached, float("nan")

        result = technique.run(paths)
        if self.cache_scores:
            np.save(cache_path, result.probabilities)
        return result.probabilities, result.ms_per_image

    # ------------------------------------------------------------------
    # Treinamento
    # ------------------------------------------------------------------

    def fit_techniques(self, seed: int = 42) -> dict[str, str | None]:
        """Ajusta as tecnicas treinaveis; T02 e T04 tem fit() vazio.

        Retorna um mapa ``{tecnica: mensagem de erro ou None}`` de modo que a
        falha de um modulo nao interrompa o ajuste dos demais (RN07).
        """
        set_deterministic(seed)
        train_paths, train_y, _ = self._load_split("train")
        val_paths, val_y, _ = self._load_split("val")

        status: dict[str, str | None] = {}
        for technique_id, technique in self.techniques.items():
            if technique_id == "T05":
                continue      # ajustada apos os escores das demais
            try:
                if technique.trainable:
                    print(f"[fit] {technique_id}: treinando com semente {seed}...")
                    if technique_id == "T01":
                        technique.fit(train_paths, train_y, val_paths=val_paths, val_y=val_y)
                    else:
                        technique.fit(train_paths, train_y)
                else:
                    print(f"[fit] {technique_id}: modelo oficial pre-treinado, fit() vazio")
                    technique.fit(train_paths, train_y)
                status[technique_id] = None
            except Exception as error:                  # noqa: BLE001
                print(f"[fit] {technique_id}: FALHOU - {error}")
                status[technique_id] = str(error)
        return status

    def fit_fusion(self, protocol: str = "standard", split: str = "val") -> str | None:
        """Ajusta T05 sobre os escores de T01-T04 em uma particao.

        ``split="val"`` reproduz a calibracao in-distribution descrita na Secao
        3.5.2. ``split="fusion"`` usa o conjunto de calibracao reservado, com um
        gerador nao visto no treinamento -- ver ``calibracao`` em
        docs/DECISOES_METODOLOGICAS.md para a justificativa empirica.
        """
        fusion = self.techniques.get("T05")
        if fusion is None:
            return "T05 nao registrada"

        try:
            paths, y, _ = self._load_split(split)
        except ValueError as error:
            return str(error)

        scores: dict[str, np.ndarray] = {}
        for technique_id in SOURCE_TECHNIQUES:
            technique = self.techniques.get(technique_id)
            if technique is None:
                continue
            try:
                scores[technique_id], _ = self._infer(
                    technique, paths, protocol, technique_id, split
                )
            except (TechniqueError, Exception) as error:     # noqa: BLE001
                print(f"[fit T05] escores de {technique_id} indisponiveis: {error}")

        if not scores:
            return f"nenhuma tecnica-fonte produziu escores na particao '{split}'"

        matrix = build_score_matrix(scores, len(paths))
        try:
            fusion.fit_scores(matrix, y)
            print(
                f"[fit T05] calibrada sobre '{split}': {len(paths)} amostras, "
                f"{len(scores)} fontes ({', '.join(sorted(scores))})"
            )
            return None
        except Exception as error:                      # noqa: BLE001
            return str(error)

    # ------------------------------------------------------------------
    # Protocolos
    # ------------------------------------------------------------------

    def run_protocol(
        self,
        protocol: str,
        split: str = "test",
        condition: str = "clean",
        paths: Sequence[Path] | None = None,
        y: np.ndarray | None = None,
        generators: np.ndarray | None = None,
        seed: int | None = None,
    ) -> list[ProtocolResult]:
        """Avalia todas as tecnicas registradas sobre um conjunto."""
        if paths is None:
            paths, y, generators = self._load_split(split)

        results: list[ProtocolResult] = []
        source_scores: dict[str, np.ndarray] = {}

        for technique_id in SOURCE_TECHNIQUES:
            technique = self.techniques.get(technique_id)
            if technique is None:
                continue
            try:
                probabilities, ms_per_image = self._infer(
                    technique, paths, protocol, technique_id, condition
                )
                source_scores[technique_id] = probabilities
                metrics = compute_metrics(y, probabilities)
                metrics["ms_per_image"] = ms_per_image       # RNF01
                result = ProtocolResult(
                    protocol=protocol,
                    technique_id=technique_id,
                    condition=condition,
                    metrics=metrics,
                    per_generator=metrics_per_generator(y, probabilities, generators),
                    probabilities=probabilities,
                    seed=seed,
                )
            except Exception as error:                  # noqa: BLE001
                # RN07/RNF04: registra a falha e segue para a proxima tecnica.
                print(f"[{protocol}/{condition}] {technique_id}: FALHOU - {error}")
                result = ProtocolResult(
                    protocol=protocol, technique_id=technique_id,
                    condition=condition, error=str(error), seed=seed,
                )
            results.append(result)

        # T05 depende dos escores das demais e e executada por ultimo.
        fusion = self.techniques.get("T05")
        if fusion is not None:
            try:
                matrix = build_score_matrix(source_scores, len(paths))
                started = time.perf_counter()
                probabilities = fusion.predict_proba_scores(matrix)
                elapsed = time.perf_counter() - started

                metrics = compute_metrics(y, probabilities)
                # Custo proprio da fusao; nao inclui a inferencia de T01-T04,
                # ja contabilizada em cada tecnica.
                metrics["ms_per_image"] = 1000.0 * elapsed / max(len(paths), 1)
                results.append(ProtocolResult(
                    protocol=protocol, technique_id="T05", condition=condition,
                    metrics=metrics,
                    per_generator=metrics_per_generator(y, probabilities, generators),
                    probabilities=probabilities, seed=seed,
                ))
            except Exception as error:                  # noqa: BLE001
                print(f"[{protocol}/{condition}] T05: FALHOU - {error}")
                results.append(ProtocolResult(
                    protocol=protocol, technique_id="T05",
                    condition=condition, error=str(error), seed=seed,
                ))

        self.results.extend(results)
        return results

    def run_robustness(self, workdir: Path, limit: int | None = None) -> list[ProtocolResult]:
        """Protocolo de robustez: recalcula as metricas sob cada perturbacao."""
        paths, y, generators = self._load_split("test")
        if limit is not None:
            # Amostragem estratificada para reduzir o custo: as perturbacoes
            # multiplicam o conjunto de teste por nove condicoes.
            rng = np.random.default_rng(42)
            selected = np.concatenate([
                rng.choice(np.flatnonzero(y == label), size=min(limit // 2, int((y == label).sum())), replace=False)
                for label in (0, 1)
            ])
            paths = [paths[i] for i in selected]
            y, generators = y[selected], generators[selected]

        workdir = Path(workdir)
        results: list[ProtocolResult] = []

        for perturbation in iter_perturbations():
            print(f"[robustez] condicao: {perturbation.name}")
            condition_dir = workdir / perturbation.name
            perturbed = materialize(paths, perturbation, condition_dir)
            results.extend(self.run_protocol(
                protocol="robustness",
                condition=perturbation.name,
                paths=perturbed, y=y, generators=generators,
            ))
        return results

    def run_multi_seed(self, technique_id: str = "T01", seeds: Sequence[int] = TRAINING_SEEDS) -> list[ProtocolResult]:
        """Repete o treinamento de uma tecnica com multiplas inicializacoes.

        O treinamento de T01 e executado com tres inicializacoes aleatorias
        distintas (sementes 42, 123 e 456) e os resultados sao reportados como
        media e desvio padrao.
        """
        technique = self.techniques.get(technique_id)
        if technique is None or not technique.trainable:
            raise ValueError(f"{technique_id} nao e treinavel")

        train_paths, train_y, _ = self._load_split("train")
        val_paths, val_y, _ = self._load_split("val")
        test_paths, test_y, test_generators = self._load_split("test")

        results = []
        for seed in seeds:
            set_deterministic(seed)
            technique.seed = seed
            if technique_id == "T01":
                technique.fit(train_paths, train_y, val_paths=val_paths, val_y=val_y)
            else:
                technique.fit(train_paths, train_y)

            probabilities = technique.run(test_paths).probabilities
            results.append(ProtocolResult(
                protocol="standard_multiseed", technique_id=technique_id,
                condition="clean", seed=seed,
                metrics=compute_metrics(test_y, probabilities),
                per_generator=metrics_per_generator(test_y, probabilities, test_generators),
                probabilities=probabilities,
            ))
        self.results.extend(results)
        return results

    # ------------------------------------------------------------------
    # Persistencia
    # ------------------------------------------------------------------

    def save(self, prefix: str = "resultados") -> dict[str, Path]:
        """Grava metricas agregadas e desagregadas em CSV."""
        import csv

        timestamp = time.strftime("%Y%m%d_%H%M%S")
        summary_path = self.results_dir / f"{prefix}_{timestamp}.csv"
        generator_path = self.results_dir / f"{prefix}_por_gerador_{timestamp}.csv"

        rows = [result.to_row() for result in self.results]
        if rows:
            columns = sorted({key for row in rows for key in row})
            # Colunas de identificacao primeiro, para leitura direta do CSV.
            leading = ["protocol", "technique", "technique_name", "condition", "seed", "status"]
            ordered = leading + [c for c in columns if c not in leading]
            with summary_path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=ordered, extrasaction="ignore")
                writer.writeheader()
                writer.writerows(rows)

        generator_rows = []
        for result in self.results:
            for generator, metrics in result.per_generator.items():
                generator_rows.append({
                    "protocol": result.protocol, "technique": result.technique_id,
                    "condition": result.condition, "seed": result.seed or "",
                    "generator": generator, **metrics,
                })
        if generator_rows:
            columns = sorted({key for row in generator_rows for key in row})
            leading = ["protocol", "technique", "condition", "seed", "generator"]
            ordered = leading + [c for c in columns if c not in leading]
            with generator_path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=ordered, extrasaction="ignore")
                writer.writeheader()
                writer.writerows(generator_rows)

        # Probabilidades brutas: necessarias para curvas ROC, teste de DeLong e
        # analise de erros sem repetir a inferencia.
        raw_dir = self.results_dir / "probabilidades"
        raw_dir.mkdir(parents=True, exist_ok=True)
        for result in self.results:
            if result.probabilities is not None:
                name = f"{result.protocol}__{result.technique_id}__{result.condition}"
                if result.seed is not None:
                    name += f"__seed{result.seed}"
                np.save(raw_dir / f"{name}.npy", result.probabilities)

        manifest = {
            "timestamp": timestamp,
            "n_results": len(self.results),
            "failures": [
                {"protocol": r.protocol, "technique": r.technique_id,
                 "condition": r.condition, "error": r.error}
                for r in self.results if r.failed
            ],
        }
        (self.results_dir / f"{prefix}_execucao_{timestamp}.json").write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        return {"summary": summary_path, "per_generator": generator_path}
