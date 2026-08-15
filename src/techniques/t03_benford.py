"""T03 - Lei de Benford sobre coeficientes DCT (Bonettini et al., 2020).

Para cada fator de qualidade JPEG do conjunto {80, 85, 90, 95, 100}, a imagem e
recomprimida e os coeficientes DCT quantizados de blocos 8 x 8 sao extraidos
para as nove primeiras frequencias AC em ordem de zigue-zague. Para cada
frequencia, a distribuicao do primeiro digito significativo e comparada a
distribuicao prevista pela lei de Benford generalizada por meio de tres
divergencias (Jensen-Shannon, Renyi e Tsallis).

O vetor de caracteristicas resulta da concatenacao das tres divergencias sobre
todas as combinacoes de bases, frequencias e fatores de qualidade, totalizando
4 x 9 x 5 x 3 = 540 dimensoes, e alimenta um classificador Random Forest.

Esta implementacao substitui a versao preliminar reportada na Secao 3.6.4 do
TCC 2 (uma unica base, um unico fator de qualidade e apenas a divergencia de
Jensen-Shannon, resultando em nove caracteristicas), que obteve AUC de 0,611.
"""

from __future__ import annotations

import io
import pickle
from pathlib import Path
from typing import Sequence

import numpy as np
from PIL import Image
from scipy.fftpack import dct

from ..config import T03, T03Config
from .base import BaseTechnique, TechniqueError

# Ordem de zigue-zague de um bloco 8 x 8. O indice 0 e o coeficiente DC; as
# nove primeiras frequencias AC correspondem as posicoes 1 a 9.
ZIGZAG_ORDER = np.array([
     0,  1,  8, 16,  9,  2,  3, 10,
    17, 24, 32, 25, 18, 11,  4,  5,
    12, 19, 26, 33, 40, 48, 41, 34,
    27, 20, 13,  6,  7, 14, 21, 28,
    35, 42, 49, 56, 57, 50, 43, 36,
    29, 22, 15, 23, 30, 37, 44, 51,
    58, 59, 52, 45, 38, 31, 39, 46,
    53, 60, 61, 54, 47, 55, 62, 63,
])

_EPS = 1e-12


# ---------------------------------------------------------------------------
# Lei de Benford generalizada
# ---------------------------------------------------------------------------


def benford_reference(base: int) -> np.ndarray:
    """Distribuicao esperada do primeiro digito na base ``base``.

    Para digitos d = 1, ..., base-1, a lei de Benford generalizada estabelece
    P(d) = log_base(1 + 1/d).
    """
    if base < 3:
        raise ValueError("a base deve ser >= 3 para produzir ao menos dois digitos")
    digits = np.arange(1, base, dtype=np.float64)
    return np.log(1.0 + 1.0 / digits) / np.log(base)


def first_digit(values: np.ndarray, base: int) -> np.ndarray:
    """Primeiro digito significativo de cada valor na base indicada.

    Valores nulos nao possuem digito significativo e devem ser removidos pelo
    chamador antes da chamada.
    """
    magnitudes = np.abs(values.astype(np.float64))
    if np.any(magnitudes == 0):
        raise ValueError("valores nulos nao possuem primeiro digito significativo")
    # Divide sucessivamente pela base ate restar um unico digito.
    exponent = np.floor(np.log(magnitudes) / np.log(base))
    digits = np.floor(magnitudes / np.power(float(base), exponent))
    # Corrige erros de arredondamento de ponto flutuante nas bordas.
    return np.clip(digits, 1, base - 1).astype(np.int64)


def first_digit_distribution(values: np.ndarray, base: int) -> np.ndarray:
    """Histograma normalizado do primeiro digito, com base-1 posicoes."""
    nonzero = values[values != 0]
    if nonzero.size == 0:
        # Sem coeficientes significativos, adota-se a distribuicao uniforme,
        # que representa ausencia de evidencia em vez de conformidade perfeita.
        return np.full(base - 1, 1.0 / (base - 1))
    digits = first_digit(nonzero, base)
    counts = np.bincount(digits, minlength=base)[1:base]
    return counts.astype(np.float64) / counts.sum()


# ---------------------------------------------------------------------------
# Divergencias
# ---------------------------------------------------------------------------


def jensen_shannon_divergence(p: np.ndarray, q: np.ndarray) -> float:
    """Divergencia de Jensen-Shannon (base 2), limitada ao intervalo [0, 1]."""
    p = np.clip(p, _EPS, None)
    q = np.clip(q, _EPS, None)
    m = 0.5 * (p + q)
    kl_pm = np.sum(p * np.log2(p / m))
    kl_qm = np.sum(q * np.log2(q / m))
    return float(0.5 * (kl_pm + kl_qm))


def renyi_divergence(p: np.ndarray, q: np.ndarray, alpha: float = T03.renyi_alpha) -> float:
    """Divergencia de Renyi de ordem alpha (alpha > 0, alpha != 1)."""
    if alpha <= 0 or np.isclose(alpha, 1.0):
        raise ValueError("alpha deve ser positivo e diferente de 1")
    p = np.clip(p, _EPS, None)
    q = np.clip(q, _EPS, None)
    total = np.sum(np.power(p, alpha) * np.power(q, 1.0 - alpha))
    return float(np.log(np.clip(total, _EPS, None)) / (alpha - 1.0))


def tsallis_divergence(p: np.ndarray, q: np.ndarray, q_order: float = T03.tsallis_q) -> float:
    """Divergencia de Tsallis de ordem q (q > 0, q != 1)."""
    if q_order <= 0 or np.isclose(q_order, 1.0):
        raise ValueError("q deve ser positivo e diferente de 1")
    p = np.clip(p, _EPS, None)
    q = np.clip(q, _EPS, None)
    total = np.sum(np.power(p, q_order) * np.power(q, 1.0 - q_order))
    return float((total - 1.0) / (q_order - 1.0))


DIVERGENCE_FUNCTIONS = {
    "jensen_shannon": jensen_shannon_divergence,
    "renyi": renyi_divergence,
    "tsallis": tsallis_divergence,
}


# ---------------------------------------------------------------------------
# Extracao de coeficientes DCT quantizados
# ---------------------------------------------------------------------------


def quantized_dct_coefficients(image: Image.Image, quality: int) -> tuple[np.ndarray, np.ndarray]:
    """Coeficientes DCT quantizados do canal de luminancia.

    A imagem e recomprimida em JPEG com o fator de qualidade indicado. A DCT-II
    bidimensional e aplicada sobre os blocos 8 x 8 do canal Y decodificado e o
    resultado e dividido pela tabela de quantizacao efetivamente gravada no
    arquivo, recuperando os coeficientes quantizados.

    Retorna ``(coeficientes, tabela)``, com coeficientes de forma
    (n_blocos, 64) na ordem raster do bloco.
    """
    buffer = io.BytesIO()
    image.convert("RGB").save(buffer, format="JPEG", quality=int(quality), subsampling=0)
    buffer.seek(0)

    with Image.open(buffer) as compressed:
        compressed.load()
        # Tabela de luminancia tal como gravada pelo codificador; usar a tabela
        # real evita divergencia entre o fator solicitado e o aplicado.
        tables = getattr(compressed, "quantization", {}) or {}
        luminance_table = tables.get(0)
        luma = np.asarray(compressed.convert("YCbCr"), dtype=np.float64)[:, :, 0]

    if luminance_table is None:
        raise TechniqueError("T03: tabela de quantizacao ausente no arquivo JPEG")
    quant_table = np.asarray(luminance_table, dtype=np.float64).reshape(8, 8)

    # Descarta as bordas que nao completam um bloco 8 x 8.
    height = (luma.shape[0] // 8) * 8
    width = (luma.shape[1] // 8) * 8
    if height == 0 or width == 0:
        raise TechniqueError("T03: imagem menor que um bloco 8 x 8")
    luma = luma[:height, :width] - 128.0   # deslocamento de nivel do JPEG

    # Reorganiza em blocos: (n_blocos, 8, 8).
    blocks = (
        luma.reshape(height // 8, 8, width // 8, 8)
        .transpose(0, 2, 1, 3)
        .reshape(-1, 8, 8)
    )
    # DCT-II ortonormal aplicada nas duas dimensoes do bloco.
    coefficients = dct(dct(blocks, axis=1, norm="ortho"), axis=2, norm="ortho")
    quantized = np.round(coefficients / quant_table)
    return quantized.reshape(-1, 64), quant_table


# ---------------------------------------------------------------------------
# Vetor de caracteristicas
# ---------------------------------------------------------------------------


def benford_features(image: Image.Image, config: T03Config = T03) -> np.ndarray:
    """Vetor de 540 divergencias em relacao a lei de Benford.

    A ordem das dimensoes e (fator de qualidade, frequencia, base, divergencia),
    percorrida do indice mais externo para o mais interno, e permanece fixa para
    que a importancia de atributos do Random Forest seja interpretavel.
    """
    references = {base: benford_reference(base) for base in config.bases}
    ac_positions = ZIGZAG_ORDER[1: config.n_frequencies + 1]

    features: list[float] = []
    for quality in config.quality_factors:
        coefficients, _ = quantized_dct_coefficients(image, quality)
        for position in ac_positions:
            frequency_values = coefficients[:, position]
            for base in config.bases:
                observed = first_digit_distribution(frequency_values, base)
                expected = references[base]
                for name in config.divergences:
                    function = DIVERGENCE_FUNCTIONS[name]
                    if name == "renyi":
                        value = function(observed, expected, config.renyi_alpha)
                    elif name == "tsallis":
                        value = function(observed, expected, config.tsallis_q)
                    else:
                        value = function(observed, expected)
                    features.append(value)

    vector = np.asarray(features, dtype=np.float64)
    if vector.size != config.n_features:
        raise TechniqueError(
            f"T03: esperadas {config.n_features} caracteristicas, obtidas {vector.size}"
        )
    # Divergencias sao finitas por construcao; NaN indicaria erro numerico.
    return np.nan_to_num(vector, nan=0.0, posinf=0.0, neginf=0.0)


def feature_names(config: T03Config = T03) -> list[str]:
    """Rotulos das 540 dimensoes, na mesma ordem de ``benford_features``."""
    names = []
    for quality in config.quality_factors:
        for index in range(config.n_frequencies):
            for base in config.bases:
                for divergence in config.divergences:
                    names.append(f"qf{quality}_ac{index + 1}_b{base}_{divergence}")
    return names


# ---------------------------------------------------------------------------
# Tecnica
# ---------------------------------------------------------------------------


class T03Benford(BaseTechnique):
    technique_id = "T03"
    trainable = True

    def __init__(self, device: str = "cpu", config: T03Config = T03, seed: int = 42, n_jobs: int = -1) -> None:
        super().__init__(device="cpu")   # extracao e classificacao operam em CPU
        self.config = config
        self.seed = seed
        self.n_jobs = n_jobs
        self.classifier = None
        self.best_params_: dict = {}

    def extract_features(self, paths: Sequence[Path]) -> np.ndarray:
        vectors = []
        for path in paths:
            with Image.open(path) as image:
                vectors.append(benford_features(image, self.config))
        return np.stack(vectors, axis=0)

    def fit(
        self,
        paths: Sequence[Path],
        y: np.ndarray,
        features: np.ndarray | None = None,
        class_weight: str | None = None,
    ) -> "T03Benford":
        """Ajusta o Random Forest com busca em grade e validacao cruzada
        estratificada de cinco particoes, adotando a AUC como metrica de
        selecao.

        ``features`` permite reaproveitar caracteristicas ja extraidas,
        evitando repetir a etapa mais custosa do pipeline.
        """
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.model_selection import GridSearchCV, StratifiedKFold

        X = self.extract_features(paths) if features is None else np.asarray(features)
        y = np.asarray(y)

        grid = {
            "n_estimators": list(self.config.grid_n_estimators),
            "max_depth": list(self.config.grid_max_depth),
        }
        search = GridSearchCV(
            RandomForestClassifier(
                random_state=self.seed,
                n_jobs=self.n_jobs,
                class_weight=class_weight,
            ),
            param_grid=grid,
            scoring="roc_auc",
            cv=StratifiedKFold(n_splits=self.config.cv_folds, shuffle=True, random_state=self.seed),
            n_jobs=1,          # o paralelismo ja ocorre dentro da floresta
            refit=True,
        )
        search.fit(X, y)

        self.classifier = search.best_estimator_
        self.best_params_ = dict(search.best_params_)
        self.best_params_["cv_auc"] = float(search.best_score_)
        self._fitted = True
        return self

    def predict_proba(self, paths: Sequence[Path], features: np.ndarray | None = None) -> np.ndarray:
        self._check_fitted()
        X = self.extract_features(paths) if features is None else np.asarray(features)
        # Coluna 1 corresponde a classe sintetica (LABEL_FAKE = 1).
        return self.classifier.predict_proba(X)[:, 1].astype(np.float64)

    def feature_importances(self) -> dict[str, float]:
        """Importancia relativa de cada dimensao, para a analise qualitativa."""
        self._check_fitted()
        return dict(zip(feature_names(self.config), self.classifier.feature_importances_))

    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as handle:
            pickle.dump({"classifier": self.classifier, "best_params": self.best_params_}, handle)

    def load(self, path: Path) -> "T03Benford":
        path = Path(path)
        if not path.exists():
            raise TechniqueError(f"T03: modelo nao encontrado em {path}")
        with path.open("rb") as handle:
            payload = pickle.load(handle)
        self.classifier = payload["classifier"]
        self.best_params_ = payload.get("best_params", {})
        self._fitted = True
        return self
