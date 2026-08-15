"""Metricas de avaliacao (Secao 3.5.1, "Metricas de avaliacao").

Metricas primarias, aplicadas a todas as tecnicas para comparacao direta:
acuracia e AUC-ROC. Metricas secundarias, aplicadas conforme a natureza de
cada tecnica: F1-Score, matriz de confusao, FPR, FNR e tempo de inferencia.

Para fins de avaliacao experimental adota-se limiar de decisao fixo de 0,5 na
conversao das probabilidades estimadas em rotulos binarios. Esse procedimento
e aplicado exclusivamente no contexto experimental e nao corresponde ao
comportamento do sistema em operacao, que exibe apenas probabilidades sem
emitir classificacao automatica (RN04).
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    roc_auc_score,
    roc_curve,
)

from .config import DECISION_THRESHOLD, LABEL_FAKE, LABEL_REAL


def compute_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float = DECISION_THRESHOLD,
) -> dict[str, float]:
    """Metricas primarias e secundarias para um conjunto rotulado.

    ``y_true`` usa 0 para imagem real e 1 para imagem sintetica.
    ``y_prob`` contem P(sintetica) em [0, 1].
    """
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob, dtype=np.float64)

    if y_true.shape != y_prob.shape:
        raise ValueError(f"formas incompativeis: {y_true.shape} e {y_prob.shape}")
    if y_true.size == 0:
        raise ValueError("conjunto de avaliacao vazio")

    y_pred = (y_prob >= threshold).astype(int)

    metrics: dict[str, float] = {
        "n_samples": float(y_true.size),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred, pos_label=LABEL_FAKE, zero_division=0)),
    }

    # A AUC exige as duas classes presentes; em avaliacoes desagregadas por
    # gerador o subconjunto pode conter apenas imagens sinteticas.
    if len(np.unique(y_true)) < 2:
        metrics["auc"] = float("nan")
    else:
        metrics["auc"] = float(roc_auc_score(y_true, y_prob))

    matrix = confusion_matrix(y_true, y_pred, labels=[LABEL_REAL, LABEL_FAKE])
    tn, fp, fn, tp = matrix.ravel()
    metrics.update({
        "tn": float(tn), "fp": float(fp), "fn": float(fn), "tp": float(tp),
        # FPR: imagens reais classificadas erroneamente como sinteticas.
        "fpr": float(fp / (fp + tn)) if (fp + tn) > 0 else float("nan"),
        # FNR: imagens sinteticas nao detectadas, relevante no contexto forense.
        "fnr": float(fn / (fn + tp)) if (fn + tp) > 0 else float("nan"),
        "precision": float(tp / (tp + fp)) if (tp + fp) > 0 else float("nan"),
        "recall": float(tp / (tp + fn)) if (tp + fn) > 0 else float("nan"),
    })
    return metrics


def metrics_per_generator(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    generators: np.ndarray,
    threshold: float = DECISION_THRESHOLD,
) -> dict[str, dict[str, float]]:
    """Avaliacao desagregada por gerador (Secao 3.5.1).

    Cada gerador e avaliado contra o conjunto completo de imagens reais, de modo
    que a AUC seja definida mesmo quando o subconjunto sintetico e de um unico
    gerador. Isso permite identificar quais modelos generativos apresentam maior
    resistencia a deteccao por cada tecnica.
    """
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob, dtype=np.float64)
    generators = np.asarray(generators)

    real_mask = y_true == LABEL_REAL
    results: dict[str, dict[str, float]] = {}

    for generator in sorted({g for g in generators[y_true == LABEL_FAKE]}):
        mask = real_mask | ((y_true == LABEL_FAKE) & (generators == generator))
        results[str(generator)] = compute_metrics(y_true[mask], y_prob[mask], threshold)
    return results


def roc_points(y_true: np.ndarray, y_prob: np.ndarray) -> dict[str, np.ndarray]:
    """Pontos da curva ROC para os graficos da Etapa 5."""
    fpr, tpr, thresholds = roc_curve(np.asarray(y_true).astype(int), np.asarray(y_prob))
    return {"fpr": fpr, "tpr": tpr, "thresholds": thresholds}


def bootstrap_auc_ci(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_resamples: int = 1000,
    confidence: float = 0.95,
    seed: int = 42,
) -> tuple[float, float]:
    """Intervalo de confianca da AUC por reamostragem bootstrap estratificada.

    Permite verificar se a diferenca de AUC entre duas tecnicas -- em especial
    entre T05 e a melhor tecnica isolada -- excede a variabilidade amostral.
    """
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob, dtype=np.float64)
    rng = np.random.default_rng(seed)

    real_idx = np.flatnonzero(y_true == LABEL_REAL)
    fake_idx = np.flatnonzero(y_true == LABEL_FAKE)
    if real_idx.size == 0 or fake_idx.size == 0:
        return float("nan"), float("nan")

    values = []
    for _ in range(n_resamples):
        # Reamostra cada classe separadamente para preservar a proporcao.
        sample = np.concatenate([
            rng.choice(real_idx, size=real_idx.size, replace=True),
            rng.choice(fake_idx, size=fake_idx.size, replace=True),
        ])
        values.append(roc_auc_score(y_true[sample], y_prob[sample]))

    alpha = (1.0 - confidence) / 2.0
    return (
        float(np.quantile(values, alpha)),
        float(np.quantile(values, 1.0 - alpha)),
    )


def delong_test(y_true: np.ndarray, prob_a: np.ndarray, prob_b: np.ndarray) -> dict[str, float]:
    """Teste de DeLong para diferenca entre duas AUCs correlacionadas.

    Aplica-se a comparacao entre tecnicas avaliadas sobre o mesmo conjunto de
    teste, situacao em que as AUCs nao sao independentes. Retorna a diferenca
    observada, a estatistica z e o valor-p bilateral.
    """
    from scipy import stats

    y_true = np.asarray(y_true).astype(int)
    positives = np.asarray(prob_a)[y_true == LABEL_FAKE], np.asarray(prob_b)[y_true == LABEL_FAKE]
    negatives = np.asarray(prob_a)[y_true == LABEL_REAL], np.asarray(prob_b)[y_true == LABEL_REAL]

    m, n = positives[0].size, negatives[0].size
    if m == 0 or n == 0:
        return {
            "auc_a": float("nan"), "auc_b": float("nan"), "difference": float("nan"),
            "z": float("nan"), "p_value": float("nan"),
        }

    def structural_components(pos: np.ndarray, neg: np.ndarray):
        # Componentes de Bamber: media de comparacoes com empate valendo 0,5.
        comparisons = (pos[:, None] > neg[None, :]).astype(float)
        comparisons += 0.5 * (pos[:, None] == neg[None, :])
        return comparisons.mean(axis=1), comparisons.mean(axis=0), comparisons.mean()

    v10_a, v01_a, auc_a = structural_components(positives[0], negatives[0])
    v10_b, v01_b, auc_b = structural_components(positives[1], negatives[1])

    s10 = np.cov(np.stack([v10_a, v10_b]), ddof=1)
    s01 = np.cov(np.stack([v01_a, v01_b]), ddof=1)
    covariance = s10 / m + s01 / n

    variance = covariance[0, 0] + covariance[1, 1] - 2 * covariance[0, 1]
    result = {
        "auc_a": float(auc_a),
        "auc_b": float(auc_b),
        "difference": float(auc_a - auc_b),
    }

    # Variancia nula ocorre quando os dois conjuntos de escores sao identicos
    # ou perfeitamente correlacionados: a diferenca e exatamente zero e o teste
    # nao se aplica.
    if variance <= 0:
        result.update({"z": float("nan"), "p_value": float("nan")})
        return result

    z = (auc_a - auc_b) / np.sqrt(variance)
    result.update({"z": float(z), "p_value": float(2 * (1 - stats.norm.cdf(abs(z))))})
    return result
