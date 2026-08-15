"""Testes das metricas por meio de conjuntos sinteticos com resultado conhecido.

Conforme a Secao 3.6.2: um classificador perfeito deve produzir acuracia, AUC e
F1 iguais a 1,0; um classificador aleatorio sobre conjunto balanceado deve
produzir acuracia e AUC proximas de 0,5. Esses testes detectam erros de
implementacao como inversao de rotulos ou uso incorreto do parametro
``average`` no F1-Score.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.metrics import (      # noqa: E402
    bootstrap_auc_ci, compute_metrics, delong_test, metrics_per_generator,
)


@pytest.fixture
def balanced_labels() -> np.ndarray:
    return np.array([0] * 100 + [1] * 100)


def test_classificador_perfeito(balanced_labels):
    probabilities = balanced_labels.astype(float)
    metrics = compute_metrics(balanced_labels, probabilities)
    assert metrics["accuracy"] == pytest.approx(1.0)
    assert metrics["auc"] == pytest.approx(1.0)
    assert metrics["f1"] == pytest.approx(1.0)
    assert metrics["fpr"] == pytest.approx(0.0)
    assert metrics["fnr"] == pytest.approx(0.0)


def test_classificador_invertido_detectado(balanced_labels):
    """Rotulos invertidos devem produzir AUC igual a 0, nao 1."""
    probabilities = 1.0 - balanced_labels.astype(float)
    metrics = compute_metrics(balanced_labels, probabilities)
    assert metrics["auc"] == pytest.approx(0.0)
    assert metrics["accuracy"] == pytest.approx(0.0)


def test_classificador_aleatorio(balanced_labels):
    rng = np.random.default_rng(42)
    probabilities = rng.uniform(size=balanced_labels.size)
    metrics = compute_metrics(balanced_labels, probabilities)
    assert metrics["auc"] == pytest.approx(0.5, abs=0.12)
    assert metrics["accuracy"] == pytest.approx(0.5, abs=0.12)


def test_matriz_de_confusao_consistente(balanced_labels):
    rng = np.random.default_rng(7)
    probabilities = rng.uniform(size=balanced_labels.size)
    metrics = compute_metrics(balanced_labels, probabilities)
    total = metrics["tn"] + metrics["fp"] + metrics["fn"] + metrics["tp"]
    assert total == balanced_labels.size
    assert metrics["accuracy"] == pytest.approx((metrics["tp"] + metrics["tn"]) / total)


def test_fpr_e_fnr_definicoes():
    # 4 reais (2 classificadas como sinteticas) e 4 sinteticas (1 nao detectada).
    y_true = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    y_prob = np.array([0.9, 0.8, 0.1, 0.2, 0.9, 0.8, 0.7, 0.1])
    metrics = compute_metrics(y_true, y_prob)
    assert metrics["fpr"] == pytest.approx(2 / 4)
    assert metrics["fnr"] == pytest.approx(1 / 4)


def test_limiar_fixo_em_meio():
    y_true = np.array([0, 1])
    # Exatamente 0,5 deve ser classificado como sintetico (>= limiar).
    metrics = compute_metrics(y_true, np.array([0.4999, 0.5]))
    assert metrics["accuracy"] == pytest.approx(1.0)


def test_auc_indefinida_com_classe_unica():
    metrics = compute_metrics(np.array([1, 1, 1]), np.array([0.9, 0.8, 0.7]))
    assert np.isnan(metrics["auc"])
    assert metrics["accuracy"] == pytest.approx(1.0)


def test_rejeita_formas_incompativeis():
    with pytest.raises(ValueError):
        compute_metrics(np.array([0, 1]), np.array([0.5]))


def test_rejeita_conjunto_vazio():
    with pytest.raises(ValueError):
        compute_metrics(np.array([]), np.array([]))


# ---------------------------------------------------------------------------
# Avaliacao desagregada
# ---------------------------------------------------------------------------


def test_metricas_por_gerador_isolam_cada_gerador():
    y_true = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    generators = np.array(["real"] * 4 + ["glide", "glide", "flux", "flux"])
    # Glide detectado com folga; Flux nao detectado.
    y_prob = np.array([0.1, 0.1, 0.2, 0.2, 0.9, 0.95, 0.05, 0.05])

    results = metrics_per_generator(y_true, y_prob, generators)
    assert set(results) == {"glide", "flux"}
    assert results["glide"]["auc"] == pytest.approx(1.0)
    assert results["flux"]["auc"] < 0.5


def test_metricas_por_gerador_usam_todas_as_reais():
    y_true = np.array([0, 0, 0, 0, 1, 1])
    generators = np.array(["real"] * 4 + ["glide", "flux"])
    y_prob = np.array([0.1, 0.2, 0.3, 0.4, 0.9, 0.9])
    results = metrics_per_generator(y_true, y_prob, generators)
    # 4 reais + 1 sintetica do gerador avaliado.
    assert results["glide"]["n_samples"] == 5


# ---------------------------------------------------------------------------
# Inferencia estatistica
# ---------------------------------------------------------------------------


def test_intervalo_bootstrap_contem_auc(balanced_labels):
    rng = np.random.default_rng(0)
    probabilities = np.where(
        balanced_labels == 1, rng.normal(0.7, 0.15, 200), rng.normal(0.3, 0.15, 200)
    ).clip(0, 1)
    observed = compute_metrics(balanced_labels, probabilities)["auc"]
    low, high = bootstrap_auc_ci(balanced_labels, probabilities, n_resamples=200)
    assert low <= observed <= high
    assert 0.0 <= low < high <= 1.0


def test_delong_sem_diferenca_para_escores_identicos(balanced_labels):
    rng = np.random.default_rng(3)
    probabilities = rng.uniform(size=balanced_labels.size)
    result = delong_test(balanced_labels, probabilities, probabilities)
    assert result["difference"] == pytest.approx(0.0)


def test_delong_detecta_diferenca(balanced_labels):
    strong = balanced_labels.astype(float) * 0.9 + 0.05
    rng = np.random.default_rng(11)
    weak = rng.uniform(size=balanced_labels.size)
    result = delong_test(balanced_labels, strong, weak)
    assert result["difference"] > 0.3
    assert result["p_value"] < 0.05
