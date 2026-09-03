"""T05 - Arquitetura hibrida por fusao tardia (contribuicao deste trabalho).

T05 combina as quatro tecnicas anteriores em uma unica decisao. Em vez de
explorar uma fonte de evidencia isolada, a arquitetura recebe os escores de
deteccao produzidos por T01, T02, T03 e T04 e os integra por meio de um
classificador de fusao, sob a hipotese de que evidencias de dominios distintos
-- espacial, espectral, estatistico e geometrico -- sao complementares e,
combinadas, elevam a capacidade discriminativa e a generalizacao para geradores
nao vistos no treinamento.

Nenhuma transformacao adicional e aplicada sobre a imagem: a tecnica opera
sobre um vetor de quatro valores. Por depender das saidas de T01 a T04, T05 e
sempre executada apos as demais.

Tratamento de escores ausentes
------------------------------
Se um modulo falha, seu escore fica indefinido. Como uma tecnica indisponivel
nao deve impedir a fusao (RN07, RNF04), o valor ausente e imputado e uma
mascara binaria de disponibilidade e concatenada ao vetor, permitindo ao
classificador distinguir "escore 0,5 observado" de "escore ausente".
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Sequence

import numpy as np

from ...configuracao import T05, T05Config
from ..base import BaseTechnique, TechniqueError

SOURCE_TECHNIQUES = ("T01", "T02", "T03", "T04")


def build_score_matrix(
    scores: dict[str, np.ndarray],
    n_samples: int,
    sources: Sequence[str] = SOURCE_TECHNIQUES,
) -> np.ndarray:
    """Organiza os escores das tecnicas-fonte em uma matriz (n, 4).

    Tecnicas ausentes do dicionario -- por indisponibilidade ou falha --
    produzem uma coluna de NaN, tratada adiante pelo imputador.
    """
    columns = []
    for technique in sources:
        values = scores.get(technique)
        if values is None:
            columns.append(np.full(n_samples, np.nan))
            continue
        values = np.asarray(values, dtype=np.float64)
        if values.shape != (n_samples,):
            raise TechniqueError(
                f"T05: escores de {technique} com forma {values.shape}, "
                f"esperado ({n_samples},)"
            )
        columns.append(values)
    return np.stack(columns, axis=1)


class T05Fusion(BaseTechnique):
    technique_id = "T05"
    trainable = True

    def __init__(self, device: str = "cpu", config: T05Config = T05, seed: int = 42) -> None:
        super().__init__(device="cpu")
        self.config = config
        self.seed = seed
        self.classifier = None
        self.scaler = None
        self.imputation_values_: np.ndarray | None = None
        self.sources = SOURCE_TECHNIQUES

    # ------------------------------------------------------------------

    def _make_classifier(self):
        from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
        from sklearn.linear_model import LogisticRegression

        if self.config.classifier == "logistic_regression":
            # Modelo linear e a escolha padrao: com apenas quatro entradas, os
            # coeficientes sao diretamente interpretaveis como o peso de cada
            # dominio de evidencia, o que sustenta a discussao dos resultados.
            return LogisticRegression(max_iter=1000, random_state=self.seed)
        if self.config.classifier == "random_forest":
            return RandomForestClassifier(n_estimators=200, random_state=self.seed, n_jobs=-1)
        if self.config.classifier == "gradient_boosting":
            return GradientBoostingClassifier(random_state=self.seed)
        raise ValueError(f"classificador de fusao desconhecido: {self.config.classifier}")

    def extract_features(self, paths: Sequence[Path]) -> np.ndarray:
        raise NotImplementedError(
            "T05 opera sobre escores, nao sobre imagens. Utilize fit_scores() e "
            "predict_proba_scores() com a matriz produzida por build_score_matrix()."
        )

    # ------------------------------------------------------------------

    def _prepare(self, matrix: np.ndarray, fit: bool) -> np.ndarray:
        matrix = np.asarray(matrix, dtype=np.float64)
        if matrix.ndim != 2 or matrix.shape[1] != len(self.sources):
            raise TechniqueError(
                f"T05: esperada matriz (n, {len(self.sources)}), obtida {matrix.shape}"
            )

        available = (~np.isnan(matrix)).astype(np.float64)

        if fit:
            if self.config.missing_strategy == "prior":
                # 0,5 representa ausencia de evidencia: nao desloca a decisao
                # em favor de nenhuma classe.
                self.imputation_values_ = np.full(matrix.shape[1], 0.5)
            elif self.config.missing_strategy == "mean":
                column_means = np.nanmean(matrix, axis=0)
                self.imputation_values_ = np.where(np.isnan(column_means), 0.5, column_means)
            else:
                raise ValueError(f"estrategia de imputacao desconhecida: {self.config.missing_strategy}")

        if self.imputation_values_ is None:
            raise TechniqueError("T05: imputacao nao definida; chame fit_scores() antes")

        filled = np.where(np.isnan(matrix), self.imputation_values_, matrix)
        return np.hstack([filled, available])

    def fit_scores(self, matrix: np.ndarray, y: np.ndarray) -> "T05Fusion":
        """Ajusta o classificador de fusao sobre os escores das demais tecnicas."""
        from sklearn.preprocessing import StandardScaler

        y = np.asarray(y)
        X = self._prepare(matrix, fit=True)

        self.scaler = StandardScaler().fit(X)
        self.classifier = self._make_classifier()
        self.classifier.fit(self.scaler.transform(X), y)
        self._fitted = True
        return self

    def fit(self, paths: Sequence[Path], y: np.ndarray, matrix: np.ndarray | None = None) -> "T05Fusion":
        if matrix is None:
            raise TechniqueError(
                "T05: fit() requer a matriz de escores de T01-T04; use fit_scores()"
            )
        return self.fit_scores(matrix, y)

    def predict_proba_scores(self, matrix: np.ndarray) -> np.ndarray:
        self._check_fitted()
        X = self.scaler.transform(self._prepare(matrix, fit=False))
        return self.classifier.predict_proba(X)[:, 1].astype(np.float64)

    def predict_proba(self, paths: Sequence[Path], matrix: np.ndarray | None = None) -> np.ndarray:
        if matrix is None:
            raise TechniqueError(
                "T05: predict_proba() requer a matriz de escores de T01-T04; "
                "use predict_proba_scores()"
            )
        return self.predict_proba_scores(matrix)

    # ------------------------------------------------------------------

    def contribution_weights(self) -> dict[str, float]:
        """Peso atribuido a cada dominio de evidencia.

        Para regressao logistica retorna os coeficientes sobre os escores
        padronizados; para modelos de arvore, a importancia de atributos. Em
        ambos os casos considera-se apenas as quatro primeiras colunas, que
        correspondem aos escores, e nao a mascara de disponibilidade.
        """
        self._check_fitted()
        if hasattr(self.classifier, "coef_"):
            values = self.classifier.coef_[0][: len(self.sources)]
        elif hasattr(self.classifier, "feature_importances_"):
            values = self.classifier.feature_importances_[: len(self.sources)]
        else:
            raise TechniqueError("T05: classificador nao expoe pesos interpretaveis")
        return {name: float(value) for name, value in zip(self.sources, values)}

    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as handle:
            pickle.dump(
                {
                    "classifier": self.classifier,
                    "scaler": self.scaler,
                    "imputation_values": self.imputation_values_,
                    "sources": self.sources,
                },
                handle,
            )

    def load(self, path: Path) -> "T05Fusion":
        path = Path(path)
        if not path.exists():
            raise TechniqueError(f"T05: modelo nao encontrado em {path}")
        with path.open("rb") as handle:
            payload = pickle.load(handle)
        self.classifier = payload["classifier"]
        self.scaler = payload["scaler"]
        self.imputation_values_ = payload["imputation_values"]
        self.sources = tuple(payload.get("sources", SOURCE_TECHNIQUES))
        self._fitted = True
        return self
