"""Geracao de figuras para a analise comparativa (Etapa 5).

Produz curvas ROC, barras de AUC por gerador, heatmaps de robustez e espectros
de magnitude. As figuras sao gravadas em PDF (vetorial, para inclusao no LaTeX)
e PNG (para inspecao rapida).
"""

from __future__ import annotations

from pathlib import Path
from typing import Mapping, Sequence

import matplotlib
matplotlib.use("Agg")           # backend sem interface grafica
import matplotlib.pyplot as plt
import numpy as np

from .config import TECHNIQUE_ORDER
from .metrics import roc_points

# Paleta com contraste suficiente em impressao monocromatica.
TECHNIQUE_COLORS = {
    "T01": "#4C72B0",
    "T02": "#DD8452",
    "T03": "#55A868",
    "T04": "#C44E52",
    "T05": "#8172B3",
}
TECHNIQUE_STYLES = {"T01": "-", "T02": "--", "T03": "-.", "T04": ":", "T05": "-"}


def _save(figure: plt.Figure, output: Path) -> Path:
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    figure.savefig(output.with_suffix(".png"), dpi=200, bbox_inches="tight")
    plt.close(figure)
    return output.with_suffix(".pdf")


def plot_roc_curves(
    y_true: np.ndarray,
    probabilities: Mapping[str, np.ndarray],
    output: Path,
    title: str = "Curvas ROC por tecnica",
) -> Path:
    """Curvas ROC sobrepostas, com a AUC de cada tecnica na legenda."""
    from sklearn.metrics import roc_auc_score

    y_true = np.asarray(y_true).astype(int)
    if len(np.unique(y_true)) < 2:
        # A curva ROC exige as duas classes; com uma so, nao ha o que plotar.
        raise ValueError(
            "curva ROC indefinida: o conjunto contem apenas a classe "
            f"{np.unique(y_true).tolist()}"
        )

    figure, axes = plt.subplots(figsize=(6.0, 5.5))
    for technique_id in TECHNIQUE_ORDER:
        scores = probabilities.get(technique_id)
        if scores is None:
            continue
        points = roc_points(y_true, scores)
        auc = roc_auc_score(y_true, scores)
        axes.plot(
            points["fpr"], points["tpr"],
            label=f"{technique_id} (AUC = {auc:.3f})",
            color=TECHNIQUE_COLORS.get(technique_id),
            linestyle=TECHNIQUE_STYLES.get(technique_id),
            linewidth=2.0 if technique_id == "T05" else 1.5,
        )

    axes.plot([0, 1], [0, 1], color="grey", linewidth=0.8, linestyle="--", label="Aleatorio")
    axes.set_xlabel("Taxa de falsos positivos (FPR)")
    axes.set_ylabel("Taxa de verdadeiros positivos (TPR)")
    axes.set_title(title)
    axes.set_xlim(0, 1)
    axes.set_ylim(0, 1.02)
    axes.legend(loc="lower right", frameon=True)
    axes.grid(alpha=0.3)
    return _save(figure, output)


def plot_auc_per_generator(
    auc_by_technique: Mapping[str, Mapping[str, float]],
    output: Path,
    title: str = "AUC por gerador",
) -> Path:
    """Barras agrupadas de AUC por gerador, uma serie por tecnica."""
    generators = sorted({g for values in auc_by_technique.values() for g in values})
    techniques = [t for t in TECHNIQUE_ORDER if t in auc_by_technique]
    if not generators or not techniques:
        raise ValueError("nenhum dado para plotar")

    figure, axes = plt.subplots(figsize=(max(8.0, 0.85 * len(generators)), 5.0))
    positions = np.arange(len(generators))
    width = 0.8 / len(techniques)

    for index, technique_id in enumerate(techniques):
        values = [auc_by_technique[technique_id].get(g, np.nan) for g in generators]
        axes.bar(
            positions + index * width - 0.4 + width / 2, values, width,
            label=technique_id, color=TECHNIQUE_COLORS.get(technique_id),
        )

    axes.axhline(0.5, color="grey", linewidth=0.8, linestyle="--")
    axes.set_xticks(positions)
    axes.set_xticklabels(generators, rotation=45, ha="right")
    axes.set_ylabel("AUC")
    axes.set_ylim(0.0, 1.05)
    axes.set_title(title)
    axes.legend(ncol=len(techniques), loc="upper center", bbox_to_anchor=(0.5, -0.28))
    axes.grid(axis="y", alpha=0.3)
    return _save(figure, output)


def plot_robustness_heatmap(
    auc_by_condition: Mapping[str, Mapping[str, float]],
    output: Path,
    title: str = "AUC sob perturbacoes (protocolo de robustez)",
) -> Path:
    """Heatmap tecnica x condicao de perturbacao."""
    techniques = [t for t in TECHNIQUE_ORDER if t in auc_by_condition]
    conditions = sorted({c for values in auc_by_condition.values() for c in values})
    # Mantem "clean" como primeira coluna, servindo de referencia visual.
    if "clean" in conditions:
        conditions = ["clean"] + [c for c in conditions if c != "clean"]

    matrix = np.array([
        [auc_by_condition[t].get(c, np.nan) for c in conditions] for t in techniques
    ])

    figure, axes = plt.subplots(figsize=(1.0 + 0.85 * len(conditions), 1.2 + 0.55 * len(techniques)))
    image = axes.imshow(matrix, cmap="RdYlGn", vmin=0.5, vmax=1.0, aspect="auto")

    axes.set_xticks(np.arange(len(conditions)))
    axes.set_xticklabels(conditions, rotation=45, ha="right")
    axes.set_yticks(np.arange(len(techniques)))
    axes.set_yticklabels(techniques)

    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            value = matrix[i, j]
            if np.isfinite(value):
                axes.text(j, i, f"{value:.3f}", ha="center", va="center", fontsize=8,
                          color="black" if value > 0.65 else "white")

    figure.colorbar(image, ax=axes, label="AUC")
    axes.set_title(title)
    return _save(figure, output)


def plot_magnitude_spectrum(image_path: Path, output: Path) -> Path:
    """Espectro de magnitude em escala logaritmica (RF05).

    A transformada e centralizada para que as baixas frequencias fiquem no
    centro da figura, evidenciando os picos periodicos frequentemente
    introduzidos pelo upsampling de modelos generativos.
    """
    from PIL import Image

    with Image.open(image_path) as image:
        luma = np.asarray(image.convert("L"), dtype=np.float64)

    spectrum = np.fft.fftshift(np.abs(np.fft.fft2(luma)))
    log_spectrum = np.log1p(spectrum)

    figure, axes = plt.subplots(1, 2, figsize=(9.0, 4.5))
    axes[0].imshow(luma, cmap="gray")
    axes[0].set_title("Imagem analisada")
    axes[0].axis("off")
    axes[1].imshow(log_spectrum, cmap="viridis")
    axes[1].set_title("Espectro de magnitude (log)")
    axes[1].axis("off")
    return _save(figure, output)


def magnitude_spectrum_array(image, size: int = 256) -> np.ndarray:
    """Espectro de magnitude normalizado em [0, 255], para exibicao na interface.

    Usa exatamente a mesma transformada de ``plot_magnitude_spectrum``,
    garantindo a consistencia exigida pelos testes de interface entre o
    espectro exibido e o aplicado na analise.
    """
    from PIL import Image

    if isinstance(image, (str, Path)):
        with Image.open(image) as handle:
            luma = np.asarray(handle.convert("L").resize((size, size)), dtype=np.float64)
    else:
        luma = np.asarray(image.convert("L").resize((size, size)), dtype=np.float64)

    spectrum = np.log1p(np.fft.fftshift(np.abs(np.fft.fft2(luma))))
    peak = spectrum.max()
    if peak > 0:
        spectrum = spectrum / peak
    return (spectrum * 255).astype(np.uint8)


def plot_training_history(history: Sequence[Mapping[str, float]], output: Path) -> Path:
    """Perda de treinamento e AUC de validacao por epoca (T01)."""
    epochs = [record["epoch"] for record in history]
    losses = [record.get("train_loss", np.nan) for record in history]
    aucs = [record.get("val_auc", np.nan) for record in history]

    figure, axis_loss = plt.subplots(figsize=(6.5, 4.0))
    axis_loss.plot(epochs, losses, color="#4C72B0", label="Perda de treinamento")
    axis_loss.set_xlabel("Epoca")
    axis_loss.set_ylabel("Perda (BCE)", color="#4C72B0")
    axis_loss.tick_params(axis="y", labelcolor="#4C72B0")
    axis_loss.grid(alpha=0.3)

    axis_auc = axis_loss.twinx()
    axis_auc.plot(epochs, aucs, color="#C44E52", label="AUC de validacao")
    axis_auc.set_ylabel("AUC de validacao", color="#C44E52")
    axis_auc.tick_params(axis="y", labelcolor="#C44E52")

    if np.isfinite(aucs).any():
        best = int(np.nanargmax(aucs))
        # Marca a epoca cujos pesos foram restaurados pela parada antecipada.
        axis_auc.axvline(epochs[best], color="grey", linestyle="--", linewidth=0.8)
        axis_auc.annotate(
            f"melhor epoca ({epochs[best]})",
            xy=(epochs[best], aucs[best]), xytext=(5, -12),
            textcoords="offset points", fontsize=8, color="grey",
        )

    axis_loss.set_title("Treinamento de T01")
    return _save(figure, output)
