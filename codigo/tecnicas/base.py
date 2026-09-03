"""Interface unificada das tecnicas de deteccao (Etapa 2 da Secao 3.5.2).

Cada tecnica e implementada como modulo independente que expoe os metodos
``fit(X_train, y_train)``, ``predict_proba(X)`` e ``score(X, y)``. Para T02 e
T04, ``fit()`` e vazio, pois ambas utilizam modelos oficiais pre-treinados e
``predict_proba()`` encapsula a chamada ao respectivo pipeline oficial de
inferencia.

A independencia entre os modulos garante que a falha isolada de um deles nao
comprometa a execucao dos demais (RN07, RNF04): o executor de experimentos e a
interface capturam ``TechniqueError`` e seguem com as tecnicas restantes.
"""

from __future__ import annotations

import abc
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import numpy as np


class TechniqueError(RuntimeError):
    """Falha na execucao de um modulo de analise.

    Sinaliza ao chamador que o resultado daquela tecnica nao esta disponivel,
    sem interromper a execucao dos demais modulos.
    """


@dataclass
class TechniqueResult:
    """Saida de uma tecnica sobre um conjunto de imagens."""

    technique_id: str
    probabilities: np.ndarray          # P(sintetica) em [0, 1], shape (n,)
    elapsed_seconds: float
    n_samples: int
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def ms_per_image(self) -> float:
        if self.n_samples == 0:
            return float("nan")
        return 1000.0 * self.elapsed_seconds / self.n_samples


class BaseTechnique(abc.ABC):
    """Contrato comum a T01-T05."""

    #: Identificador curto, por exemplo "T01".
    technique_id: str = "TXX"
    #: Se False, ``fit()`` e um no-op (modelos oficiais pre-treinados).
    trainable: bool = True

    def __init__(self, device: str = "cpu") -> None:
        self.device = device
        self._fitted = False

    # ------------------------------------------------------------------
    # Contrato
    # ------------------------------------------------------------------

    @abc.abstractmethod
    def extract_features(self, paths: Sequence[Path]) -> np.ndarray:
        """Extrai as caracteristicas especificas da tecnica.

        Recebe caminhos de imagem em vez de arrays para que tecnicas que
        dependem de recompressao JPEG (T03) ou de pipelines oficiais (T02, T04)
        possam operar sobre o arquivo original.
        """

    def fit(self, paths: Sequence[Path], y: np.ndarray) -> "BaseTechnique":
        """Ajusta a tecnica ao conjunto de treinamento."""
        if not self.trainable:
            self._fitted = True
            return self
        raise NotImplementedError

    @abc.abstractmethod
    def predict_proba(self, paths: Sequence[Path]) -> np.ndarray:
        """Retorna P(imagem sintetica) para cada caminho, em [0, 1]."""

    # ------------------------------------------------------------------
    # Utilidades compartilhadas
    # ------------------------------------------------------------------

    def score(self, paths: Sequence[Path], y: np.ndarray) -> dict[str, float]:
        """Calcula as metricas da tecnica sobre um conjunto rotulado."""
        from ..metricas import compute_metrics

        result = self.run(paths)
        metrics = compute_metrics(y, result.probabilities)
        metrics["ms_per_image"] = result.ms_per_image
        return metrics

    def run(self, paths: Sequence[Path]) -> TechniqueResult:
        """Executa a inferencia medindo o tempo total (RNF01)."""
        paths = list(paths)
        start = time.perf_counter()
        probabilities = np.asarray(self.predict_proba(paths), dtype=np.float64)
        elapsed = time.perf_counter() - start

        self._validate_probabilities(probabilities, len(paths))
        return TechniqueResult(
            technique_id=self.technique_id,
            probabilities=probabilities,
            elapsed_seconds=elapsed,
            n_samples=len(paths),
        )

    def _validate_probabilities(self, probabilities: np.ndarray, n: int) -> None:
        if probabilities.shape != (n,):
            raise TechniqueError(
                f"{self.technique_id}: esperado vetor de forma ({n},), "
                f"obtido {probabilities.shape}"
            )
        if not np.isfinite(probabilities).all():
            raise TechniqueError(
                f"{self.technique_id}: saida contem valores NaN ou infinitos"
            )
        if probabilities.min() < 0.0 or probabilities.max() > 1.0:
            raise TechniqueError(
                f"{self.technique_id}: probabilidades fora do intervalo [0, 1] "
                f"(min={probabilities.min():.4f}, max={probabilities.max():.4f})"
            )

    def _check_fitted(self) -> None:
        if not self._fitted:
            raise TechniqueError(
                f"{self.technique_id}: fit() deve ser chamado antes de predict_proba()"
            )

    # ------------------------------------------------------------------
    # Persistencia
    # ------------------------------------------------------------------

    def save(self, path: Path) -> None:
        raise NotImplementedError

    def load(self, path: Path) -> "BaseTechnique":
        raise NotImplementedError

    def __repr__(self) -> str:
        state = "ajustada" if self._fitted else "nao ajustada"
        return f"<{type(self).__name__} {self.technique_id} ({state})>"
