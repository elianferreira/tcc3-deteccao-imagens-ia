"""T02 - SPAI: deteccao por aprendizado espectral (Karageorgiou et al., 2025).

Replicacao por meio do codigo e dos pesos oficiais disponibilizados pelos
autores em https://github.com/mever-team/spai sob licenca Apache 2.0.

Conforme a Secao 3.5.1, T02 nao e treinada neste trabalho: o treinamento do
SPAI exige GPU de 48 GB, indisponivel no ambiente do projeto, e o pre-treino
espectral e o componente determinante do metodo. Adota-se o checkpoint oficial,
pre-treinado pelos autores sobre o dataset de Corvi et al. (2024), em modo de
inferencia. A inferencia requer menos de 8 GB de memoria de GPU.

O repositorio oficial exige Python 3.11 e PyTorch com CUDA 12.4, versoes
distintas das adotadas neste projeto. Por essa razao a integracao ocorre por
subprocesso, sobre um ambiente virtual separado, e nao por importacao direta:
isso isola as dependencias conflitantes e evita que a instalacao do SPAI
interfira nas demais tecnicas (risco R02).
"""

from __future__ import annotations

import csv
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Sequence

import numpy as np

from ...configuracao import EXTERNAL, ExternalConfig
from ..base import BaseTechnique, TechniqueError

# Nomes de coluna aceitos no CSV de saida do pipeline oficial. O repositorio
# nao fixa o rotulo da coluna de escore entre versoes, entao aceitamos os
# nomes conhecidos e falhamos de forma explicita se nenhum estiver presente.
# "spai" e o nome efetivamente emitido pela versao atual do pipeline oficial;
# os demais cobrem variacoes conhecidas entre versoes.
_SCORE_COLUMNS = ("spai", "score", "prediction", "probability", "pred", "spai_score", "output")
_PATH_COLUMNS = ("image", "path", "filename", "file", "image_path")

# O CSVDataset oficial aceita apenas train/val/test e filtra as linhas por
# esse valor; "test" e o unico coerente com inferencia.
_SPLIT = "test"


class T02SPAI(BaseTechnique):
    technique_id = "T02"
    trainable = False   # fit() e vazio: modelo oficial pre-treinado

    def __init__(
        self,
        device: str = "cuda",
        external: ExternalConfig = EXTERNAL,
        python_executable: str | None = None,
        timeout: int | None = None,
        batch_size: int = 1,
        seconds_per_image: float = 3.0,
        min_timeout: int = 1800,
    ) -> None:
        super().__init__(device=device)
        self.external = external
        self.batch_size = batch_size
        # Um limite fixo nao serve: a inferencia custa cerca de 0,34 s por
        # imagem no hardware de referencia, de modo que um teto de uma hora
        # interrompe qualquer conjunto acima de ~10.000 imagens no meio da
        # execucao. Quando `timeout` nao e informado, ele passa a ser derivado
        # do tamanho do lote, com folga de quase dez vezes sobre o custo medido
        # para absorver GPUs mais lentas e a carga do sistema.
        self.timeout = timeout
        self.seconds_per_image = seconds_per_image
        self.min_timeout = min_timeout
        # Interpretador do ambiente virtual do SPAI (Python 3.11). Por padrao
        # procura um venv dentro do proprio repositorio clonado.
        self.python_executable = python_executable or self._default_interpreter()
        self._fitted = True   # nao ha ajuste a realizar

    def _timeout_for(self, n_images: int) -> int:
        """Limite de tempo para um lote de ``n_images``."""
        if self.timeout is not None:
            return self.timeout
        return max(self.min_timeout, int(self.seconds_per_image * n_images))

    def _default_interpreter(self) -> str:
        repo = self.external.spai_repo
        candidates = [
            repo / ".venv" / "Scripts" / "python.exe",   # Windows
            repo / ".venv" / "bin" / "python",           # Linux/macOS
        ]
        for candidate in candidates:
            if candidate.exists():
                return str(candidate)
        return sys.executable

    # ------------------------------------------------------------------

    def is_available(self) -> tuple[bool, str]:
        """Verifica se repositorio e checkpoint estao presentes.

        Retorna ``(disponivel, motivo)`` para que o chamador possa registrar a
        indisponibilidade sem interromper as demais tecnicas (RN07).
        """
        if not self.external.spai_repo.exists():
            return False, (
                f"repositorio oficial ausente em {self.external.spai_repo}; "
                "execute automacao/setup_external.py"
            )
        if not self.external.spai_checkpoint.exists():
            return False, (
                f"checkpoint oficial ausente em {self.external.spai_checkpoint}; "
                "baixe os pesos indicados no README do repositorio oficial"
            )
        if not Path(self.python_executable).exists() and shutil.which(self.python_executable) is None:
            return False, f"interpretador nao encontrado: {self.python_executable}"
        return True, "ok"

    def extract_features(self, paths: Sequence[Path]) -> np.ndarray:
        """O pre-processamento e realizado pelo proprio pipeline oficial do
        SPAI, que opera diretamente sobre a imagem de entrada; nenhuma
        transformacao adicional e aplicada externamente."""
        raise NotImplementedError(
            "T02 nao expoe caracteristicas intermediarias: o pipeline oficial "
            "produz diretamente o escore de deteccao."
        )

    def fit(self, paths: Sequence[Path], y: np.ndarray) -> "T02SPAI":
        """No-op: T02 utiliza o checkpoint oficial pre-treinado."""
        self._fitted = True
        return self

    # ------------------------------------------------------------------

    def predict_proba(self, paths: Sequence[Path]) -> np.ndarray:
        available, reason = self.is_available()
        if not available:
            raise TechniqueError(f"T02 indisponivel: {reason}")

        paths = [Path(p).resolve() for p in paths]
        if not paths:
            return np.empty(0, dtype=np.float64)

        with tempfile.TemporaryDirectory(prefix="spai_") as workdir:
            work = Path(workdir)
            input_csv = work / "input.csv"
            output_dir = work / "out"
            output_dir.mkdir()

            # Contrato do CSVDataset oficial (spai/data/data_finetune.py):
            # exige as colunas "image" (caminho absoluto), "split" -- restrita a
            # train/val/test e usada para filtrar as linhas -- e "class", da qual
            # o numero de classes e derivado. Em inferencia o rotulo nao e
            # utilizado na predicao; grava-se 0 apenas para satisfazer o formato.
            with input_csv.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.writer(handle)
                writer.writerow(["image", "split", "class"])
                for path in paths:
                    writer.writerow([str(path), _SPLIT, 0])

            # --model precisa ser explicito e absoluto: seu valor padrao,
            # "./weights/spai.pth", e resolvido a partir do diretorio do
            # repositorio oficial, onde o checkpoint deste projeto nao esta.
            # DATA.NUM_WORKERS (23/09/2026): o padrao do SPAI e 24
            # (`externo/spai/spai/config.py:61`), dimensionado para o servidor
            # dos autores. Nesta maquina -- 15,7 GB, dos quais ~12 GB ja
            # tomados -- os 24 processos estouram a memoria e a rodada morre
            # com "DataLoader worker (pid(s) ...) exited unexpectedly", as
            # vezes precedido de "arquivo de paginacao e muito pequeno".
            #
            # Observado em 21/09 na tela v2 e em 23/09 aqui, nas duas amostras
            # de `fpr_raise1k_todas.py`. O `infer` nao expoe --data-workers (so
            # o `train`), entao o caminho e --opt, que o `get_config` repassa ao
            # `merge_from_list` do yacs.
            #
            # ⚠️ Nao use 0: DATA.PREFETCH_FACTOR e 2 e o PyTorch recusa prefetch
            # sem multiprocessing; anular os dois esbarra na checagem de tipo do
            # yacs, que nao troca int por None.
            #
            # ✅ Nao altera escore -- o numero de workers governa so o
            # paralelismo de leitura. Conferido na tela v2: 0,885062 com 24 e
            # com 2, identico ate a sexta casa.
            workers = os.environ.get("TCC3_T02_WORKERS", "2").strip() or "2"
            command = [
                self.python_executable, "-m", "spai", "infer",
                "--input", str(input_csv),
                "--output", str(output_dir),
                "--model", str(self.external.spai_checkpoint.resolve()),
                "--batch-size", str(self.batch_size),
                "--split", _SPLIT,
                "--opt", "DATA.NUM_WORKERS", workers,
            ]
            limit = self._timeout_for(len(paths))
            try:
                completed = subprocess.run(
                    command,
                    cwd=str(self.external.spai_repo),
                    capture_output=True,
                    text=True,
                    timeout=limit,
                    check=False,
                )
            except subprocess.TimeoutExpired as error:
                raise TechniqueError(
                    f"T02: inferencia de {len(paths)} imagens excedeu {limit}s"
                ) from error

            if completed.returncode != 0:
                # O pipeline oficial escreve progresso em stderr, de modo que a
                # excecao real fica soterrada. Persistir a saida integra e a
                # unica forma de diagnosticar falhas em lotes grandes.
                log_path = self._persist_failure(completed)
                tail = self._meaningful_tail(completed)
                raise TechniqueError(
                    f"T02: pipeline oficial falhou (codigo {completed.returncode}); "
                    f"saida completa em {log_path}\n{tail}"
                )

            return self._read_scores(output_dir, paths)

    # ------------------------------------------------------------------

    def _persist_failure(self, completed) -> Path:
        """Grava stdout e stderr integrais para diagnostico posterior."""
        from ...configuracao import RESULTS_DIR

        log_dir = RESULTS_DIR / "logs_t02"
        log_dir.mkdir(parents=True, exist_ok=True)
        log_path = log_dir / f"spai_falha_{int(time.time())}.log"
        log_path.write_text(
            "=== STDOUT ===\n" + (completed.stdout or "")
            + "\n\n=== STDERR ===\n" + (completed.stderr or ""),
            encoding="utf-8", errors="replace",
        )
        return log_path

    @staticmethod
    def _meaningful_tail(completed, max_chars: int = 1200) -> str:
        """Extrai as linhas de erro, ignorando o progresso rotineiro.

        As linhas de progresso do SPAI ("Test: [N/M] ...") ocupam todo o final
        do stderr e empurram a excecao para fora de qualquer recorte simples.
        """
        combined = ((completed.stderr or "") + "\n" + (completed.stdout or "")).splitlines()
        marcadores = ("Traceback", "Error", "error:", "Exception", "assert",
                      "CUDA", "out of memory", "Killed", "cannot", "No such file")
        relevantes = [
            line for line in combined
            if any(m in line for m in marcadores) and "Test: [" not in line
        ]
        if not relevantes:
            relevantes = [line for line in combined if line.strip()][-12:]
        return "\n".join(relevantes)[-max_chars:]

    def _read_scores(self, output_dir: Path, paths: Sequence[Path]) -> np.ndarray:
        csv_files = sorted(output_dir.glob("*.csv"))
        if not csv_files:
            raise TechniqueError(f"T02: nenhum CSV de predicoes gerado em {output_dir}")

        with csv_files[0].open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        if not rows:
            raise TechniqueError("T02: CSV de predicoes vazio")

        columns = rows[0].keys()
        score_column = next((c for c in _SCORE_COLUMNS if c in columns), None)
        if score_column is None:
            raise TechniqueError(
                f"T02: coluna de escore nao identificada; colunas presentes: {list(columns)}"
            )
        path_column = next((c for c in _PATH_COLUMNS if c in columns), None)

        if path_column is None:
            # Sem coluna de caminho, assume-se que a ordem foi preservada.
            if len(rows) != len(paths):
                raise TechniqueError(
                    f"T02: {len(rows)} predicoes para {len(paths)} imagens e nenhuma "
                    "coluna de caminho para realinhar"
                )
            scores = [float(row[score_column]) for row in rows]
        else:
            # Reindexa pelo nome do arquivo para nao depender da ordem de saida.
            by_name = {Path(row[path_column]).name: float(row[score_column]) for row in rows}
            missing = [p.name for p in paths if p.name not in by_name]
            if missing:
                raise TechniqueError(
                    f"T02: {len(missing)} imagens sem predicao (ex.: {missing[:3]})"
                )
            scores = [by_name[p.name] for p in paths]

        return self._to_probabilities(np.asarray(scores, dtype=np.float64))

    @staticmethod
    def _to_probabilities(scores: np.ndarray) -> np.ndarray:
        """Converte a saida do SPAI em probabilidade no intervalo [0, 1].

        O pipeline oficial pode emitir logits ou probabilidades conforme a
        versao. Valores ja contidos em [0, 1] sao mantidos; caso contrario
        aplica-se a sigmoide, preservando a ordenacao e, portanto, a AUC.
        """
        if scores.size and (scores.min() < 0.0 or scores.max() > 1.0):
            return 1.0 / (1.0 + np.exp(-scores))
        return scores
