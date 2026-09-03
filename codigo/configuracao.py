"""Configuracao central do projeto.

Concentra caminhos, constantes experimentais e hiperparametros descritos no
Capitulo 3 do TCC 2, de modo que nenhum valor experimental fique espalhado
pelos modulos. Valores podem ser sobrescritos por variaveis de ambiente para
permitir execucao em ambiente alternativo (Google Colab), conforme a Secao 3.2.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# ``data/`` mantem o nome em ingles de proposito: os manifestos gravam caminho
# absoluto de cada imagem, e sao 132.640 linhas nos dois manifestos de 30k mais
# os tres da escala. Renomear a pasta obrigaria a reescrever o registro de
# reproducao inteiro, que o .gitignore nao protege. Ver documentacao/ESTRUTURA.md.
DATA_DIR = Path(os.environ.get("TCC3_DATA_DIR", ROOT / "data"))
RESULTS_DIR = Path(os.environ.get("TCC3_RESULTS_DIR", ROOT / "resultados"))
EXTERNAL_DIR = Path(os.environ.get("TCC3_EXTERNAL_DIR", ROOT / "externo"))
WEIGHTS_DIR = Path(os.environ.get("TCC3_WEIGHTS_DIR", ROOT / "pesos"))

# Semente unica usada em random, NumPy, PyTorch e CUDA (Etapa 1, RNF02).
SEED = 42
# Sementes das tres inicializacoes independentes de T01 (Etapa 4).
TRAINING_SEEDS = (42, 123, 456)

# Resolucao do dataset de Corvi et al. (2024).
IMAGE_SIZE = 256

# Limiar fixo usado apenas na avaliacao experimental. A interface nao converte
# probabilidade em rotulo binario (RN04).
DECISION_THRESHOLD = 0.5

LABEL_REAL = 0
LABEL_FAKE = 1

# --------------------------------------------------------------------------
# Geradores do benchmark (Secao 3.5.1)
# --------------------------------------------------------------------------

DIFFUSION_GENERATORS = (
    "glide",
    "stable_diffusion_1_3",
    "stable_diffusion_1_4",
    "stable_diffusion_2",
    "stable_diffusion_xl",
    "stable_diffusion_3",
    "flux",
    "dalle2",
    "dalle3",
    "adobe_firefly",
)

# Familias mantidas fora do treinamento no protocolo OOD.
HELD_OUT_GENERATORS = (
    "gigagan",          # rede adversarial generativa
    "midjourney_v5",    # gerador comercial, arquitetura nao divulgada
    "midjourney_v6_1",
)

ALL_GENERATORS = DIFFUSION_GENERATORS + HELD_OUT_GENERATORS

# Gerador do conjunto de TREINAMENTO de Corvi et al. (2024): as 180.000
# sinteticas do corpus de treino provem de um unico modelo de difusao latente,
# distinto dos 13 geradores do benchmark de avaliacao. Manter essa separacao
# explicita permite o protocolo de generalizacao mais rigoroso: treinar em um
# gerador e avaliar nos treze restantes, nenhum visto no treinamento.
TRAINING_GENERATORS = ("latent_diffusion",)

BENCHMARK_GENERATORS = ALL_GENERATORS

# Gerador reservado exclusivamente a calibracao do classificador de fusao (T05).
# Nao participa nem do treinamento das tecnicas-base nem da avaliacao final,
# constituindo um terceiro conjunto disjunto. A justificativa empirica dessa
# separacao esta em documentacao/DECISOES_METODOLOGICAS.md.
FUSION_CALIBRATION_GENERATORS = ("glide",)

# Proporcao das imagens reais de teste desviada para a calibracao da fusao, de
# modo que o conjunto de calibracao contenha as duas classes sem reaproveitar
# imagens reais que serao usadas na avaliacao.
FUSION_CALIBRATION_REAL_RATIO = 0.30

REAL_SOURCES = ("raise", "fodb", "imagenet", "coco", "open_images")

# Particionamento estratificado por classe (Secao 3.5.1).
SPLIT_RATIOS = {"train": 0.70, "val": 0.15, "test": 0.15}

# --------------------------------------------------------------------------
# Protocolo de robustez (Secao 3.6.1)
# --------------------------------------------------------------------------

JPEG_QUALITIES = (50, 70, 85)
GAUSSIAN_SIGMAS = (1.0, 3.0, 5.0)
RESIZE_FACTORS = (0.50, 0.70, 0.85)

# --------------------------------------------------------------------------
# T01 - CNN com matrizes de coocorrencia RGB (Nataraj et al., 2019)
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class T01Config:
    batch_size: int = 40
    max_epochs: int = 50
    early_stopping_patience: int = 10   # epocas sem melhora da AUC de validacao
    learning_rate: float = 1e-4
    # Deslocamento unitario horizontal usado no calculo da coocorrencia.
    offset: tuple[int, int] = (0, 1)
    levels: int = 256


# --------------------------------------------------------------------------
# T03 - Lei de Benford sobre coeficientes DCT (Bonettini et al., 2020)
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class T03Config:
    # 4 bases x 9 frequencias x 5 fatores de qualidade x 3 divergencias = 540
    bases: tuple[int, ...] = (10, 20, 40, 60)
    n_frequencies: int = 9               # primeiras AC em ordem zigue-zague
    quality_factors: tuple[int, ...] = (80, 85, 90, 95, 100)
    divergences: tuple[str, ...] = ("jensen_shannon", "renyi", "tsallis")
    renyi_alpha: float = 0.5
    tsallis_q: float = 0.5

    # Busca em grade com validacao cruzada estratificada de 5 particoes.
    grid_n_estimators: tuple[int, ...] = (100, 200, 500)
    grid_max_depth: tuple[int | None, ...] = (10, 20, None)
    cv_folds: int = 5

    @property
    def n_features(self) -> int:
        return (
            len(self.bases)
            * self.n_frequencies
            * len(self.quality_factors)
            * len(self.divergences)
        )


# --------------------------------------------------------------------------
# T05 - Arquitetura hibrida de fusao tardia
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class T05Config:
    # Classificador de fusao sobre o vetor de 4 escores [T01, T02, T03, T04].
    classifier: str = "logistic_regression"   # ou "random_forest", "gradient_boosting"
    # Estrategia para escore ausente quando um modulo falha (RN07/RNF04).
    missing_strategy: str = "prior"           # imputa a prevalencia da classe
    cv_folds: int = 5


@dataclass(frozen=True)
class ExternalConfig:
    """Repositorios oficiais executados em modo de inferencia (T02 e T04)."""

    spai_repo: Path = field(default_factory=lambda: EXTERNAL_DIR / "spai")
    spai_checkpoint: Path = field(default_factory=lambda: WEIGHTS_DIR / "spai.pth")
    geometry_repo: Path = field(default_factory=lambda: EXTERNAL_DIR / "projective-geometry")
    geometry_weights: Path = field(default_factory=lambda: WEIGHTS_DIR / "projective_geometry")


T01 = T01Config()
T03 = T03Config()
T05 = T05Config()
EXTERNAL = ExternalConfig()

TECHNIQUE_ORDER = ("T01", "T02", "T03", "T04", "T05")

TECHNIQUE_NAMES = {
    "T01": "Coocorrencia RGB + CNN (Nataraj et al., 2019)",
    "T02": "SPAI - Aprendizado espectral (Karageorgiou et al., 2025)",
    "T03": "Lei de Benford sobre DCT (Bonettini et al., 2020)",
    "T04": "Geometria projetiva (Sarkar et al., 2024)",
    "T05": "Arquitetura hibrida (proposta)",
}


def ensure_dirs() -> None:
    for path in (DATA_DIR, RESULTS_DIR, EXTERNAL_DIR, WEIGHTS_DIR):
        path.mkdir(parents=True, exist_ok=True)
