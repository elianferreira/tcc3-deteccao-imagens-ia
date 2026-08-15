"""T04 - Deteccao por geometria projetiva (Sarkar et al., 2024).

Replicacao do metodo publicado em https://github.com/hanlinm2/projective-geometry
(CVPR 2024), que detecta imagens sinteticas a partir de inconsistencias
geometricas, sem acesso direto aos pixels da imagem.

Sao extraidas tres representacoes geometricas independentes por meio de modelos
auxiliares oficiais, cada uma com seu proprio classificador:

===================  ==========================  ==================
Representacao        Classificador               Diretorio oficial
===================  ==========================  ==================
Campo de perspectiva ResNet-50                   perspective_fields/
Segmentos de reta    PointNet                    line_segment/
Objeto-sombra        ResNet-50                   object_shadow/
===================  ==========================  ==================

Como os classificadores operam apenas sobre caracteristicas geometricas
derivadas, e nao sobre os pixels, o metodo tende a ser mais robusto a variacoes
de baixo nivel especificas de cada gerador.

A tecnica e executada em modo de inferencia. Os tres escores sao agregados em
uma unica probabilidade, tal como no trabalho original; os escores individuais
permanecem acessiveis em ``last_component_scores_`` para a analise de erros
prevista na Etapa 5.
"""

from __future__ import annotations

import csv
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Sequence

import numpy as np

from ..config import EXTERNAL, ExternalConfig
from .base import BaseTechnique, TechniqueError

COMPONENTS = ("perspective_fields", "line_segment", "object_shadow")


class T04ProjectiveGeometry(BaseTechnique):
    technique_id = "T04"
    trainable = False   # extratores e classificadores oficiais pre-treinados

    def __init__(
        self,
        device: str = "cuda",
        external: ExternalConfig = EXTERNAL,
        python_executable: str | None = None,
        components: Sequence[str] = COMPONENTS,
        aggregation: str = "mean",
        timeout: int = 3600,
    ) -> None:
        super().__init__(device=device)
        self.external = external
        self.python_executable = python_executable or self._default_interpreter()
        self.components = tuple(components)
        self.aggregation = aggregation
        self.timeout = timeout
        self.last_component_scores_: dict[str, np.ndarray] = {}
        self._fitted = True

    def _default_interpreter(self) -> str:
        repo = self.external.geometry_repo
        for candidate in (repo / ".venv" / "Scripts" / "python.exe", repo / ".venv" / "bin" / "python"):
            if candidate.exists():
                return str(candidate)
        return sys.executable

    # ------------------------------------------------------------------

    def is_available(self) -> tuple[bool, str]:
        repo = self.external.geometry_repo
        if not repo.exists():
            return False, (
                f"repositorio oficial ausente em {repo}; execute scripts/setup_external.py"
            )
        missing = [c for c in self.components if not (repo / c).exists()]
        if missing:
            return False, f"subdiretorios oficiais ausentes: {missing}"
        if not self.external.geometry_weights.exists():
            return False, (
                f"pesos dos classificadores ausentes em {self.external.geometry_weights}"
            )
        if not Path(self.python_executable).exists() and shutil.which(self.python_executable) is None:
            return False, f"interpretador nao encontrado: {self.python_executable}"

        # O bloqueio real de T04 nao e um arquivo faltando: os tres
        # classificadores oficiais nao recebem pixels. Cada um carrega uma
        # representacao geometrica ja extraida por um modelo auxiliar externo,
        # que o repositorio nao distribui. Verificado no codigo oficial:
        #
        #   perspective_fields/fields_dataset.py:31  torch.load(field_path)
        #       -> tensores .pt de PerspectiveFields (Jin et al.), via detectron2
        #   object_shadow/dataset.py:20-21           Image.open(shadow/object)
        #       -> mascaras de SSISv2, via detectron2/AdelaiDet
        #   line_segment/lines_dataset.py:22         image_path_to_lines[...]
        #       -> segmentos pre-extraidos por detector de retas
        #
        # O proprio artigo declara: "All three classifiers are denied access to
        # image pixels, and look only at derived geometric features."
        #
        # Reportar "script de inferencia ausente" seria enganoso: sugere que
        # escrever um adaptador resolveria, quando o que falta sao os
        # extratores. Ver external/CONTRATO.md.
        return False, (
            "os classificadores oficiais nao operam sobre pixels: exigem "
            "representacoes geometricas pre-extraidas (campos de perspectiva, "
            "mascaras objeto-sombra, segmentos de reta) produzidas por modelos "
            "auxiliares dependentes de detectron2, sem suporte em Windows. "
            "Ver external/CONTRATO.md"
        )

    def extract_features(self, paths: Sequence[Path]) -> np.ndarray:
        """Matriz (n, 3) com os escores das tres representacoes geometricas."""
        available, reason = self.is_available()
        if not available:
            raise TechniqueError(f"T04 indisponivel: {reason}")

        paths = [Path(p).resolve() for p in paths]
        columns = []
        failures = []
        for component in self.components:
            try:
                columns.append(self._run_component(component, paths))
            except TechniqueError as error:
                # A falha de uma representacao nao invalida as demais; o metodo
                # segue com as disponiveis, em conformidade com RN07.
                failures.append(f"{component}: {error}")
                columns.append(np.full(len(paths), np.nan))

        matrix = np.stack(columns, axis=1)
        if np.isnan(matrix).all():
            raise TechniqueError("T04: todas as representacoes falharam\n" + "\n".join(failures))

        self.last_component_scores_ = {
            component: matrix[:, index] for index, component in enumerate(self.components)
        }
        return matrix

    def _run_component(self, component: str, paths: Sequence[Path]) -> np.ndarray:
        """Executa o classificador oficial de uma representacao geometrica.

        A invocacao segue o contrato documentado em ``external/CONTRATO.md``:
        um script ``infer.py`` no diretorio da representacao que recebe um CSV
        de caminhos e grava um JSON ``{caminho: escore}``.
        """
        script = self.external.geometry_repo / component / "infer.py"
        if not script.exists():
            raise TechniqueError(
                f"script de inferencia ausente: {script}. Consulte external/CONTRATO.md "
                "para adaptar o script oficial ao contrato esperado."
            )

        with tempfile.TemporaryDirectory(prefix=f"geo_{component}_") as workdir:
            work = Path(workdir)
            input_csv = work / "input.csv"
            output_json = work / "scores.json"

            with input_csv.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.writer(handle)
                writer.writerow(["image"])
                for path in paths:
                    writer.writerow([str(path)])

            command = [
                self.python_executable, str(script),
                "--input", str(input_csv),
                "--output", str(output_json),
                "--weights", str(self.external.geometry_weights / component),
                "--device", self.device,
            ]
            completed = subprocess.run(
                command,
                cwd=str(self.external.geometry_repo),
                capture_output=True,
                text=True,
                timeout=self.timeout,
                check=False,
            )
            if completed.returncode != 0:
                tail = (completed.stderr or completed.stdout or "").strip()[-800:]
                raise TechniqueError(f"falha na execucao\n{tail}")
            if not output_json.exists():
                raise TechniqueError(f"saida nao gerada em {output_json}")

            scores = json.loads(output_json.read_text(encoding="utf-8"))

        by_name = {Path(key).name: float(value) for key, value in scores.items()}
        missing = [p.name for p in paths if p.name not in by_name]
        if missing:
            raise TechniqueError(f"{len(missing)} imagens sem escore (ex.: {missing[:3]})")
        return np.asarray([by_name[p.name] for p in paths], dtype=np.float64)

    def fit(self, paths: Sequence[Path], y: np.ndarray) -> "T04ProjectiveGeometry":
        """No-op: T04 utiliza os extratores e classificadores oficiais."""
        self._fitted = True
        return self

    # ------------------------------------------------------------------

    def predict_proba(self, paths: Sequence[Path]) -> np.ndarray:
        matrix = self.extract_features(paths)
        matrix = self._to_probabilities(matrix)

        if self.aggregation == "mean":
            # nanmean ignora representacoes indisponiveis para cada amostra.
            aggregated = np.nanmean(matrix, axis=1)
        elif self.aggregation == "max":
            aggregated = np.nanmax(matrix, axis=1)
        else:
            raise ValueError(f"agregacao desconhecida: {self.aggregation}")

        # Amostras sem nenhuma representacao valida recebem 0,5, valor neutro
        # que nao favorece nenhuma das classes.
        return np.where(np.isnan(aggregated), 0.5, aggregated)

    @staticmethod
    def _to_probabilities(matrix: np.ndarray) -> np.ndarray:
        """Normaliza cada coluna para [0, 1], preservando a ordenacao."""
        output = matrix.copy()
        for index in range(output.shape[1]):
            column = output[:, index]
            finite = column[np.isfinite(column)]
            if finite.size and (finite.min() < 0.0 or finite.max() > 1.0):
                output[:, index] = 1.0 / (1.0 + np.exp(-column))
        return output
