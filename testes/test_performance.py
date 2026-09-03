"""Testes de desempenho (Secao 3.6.2).

O tempo de inferencia por imagem e medido em CPU, com calculo de media e desvio
padrao. O criterio de aprovacao e tempo medio inferior a 30 segundos por imagem
para todas as tecnicas, conforme RNF01.

Os testes usam um lote reduzido: a Secao 3.6.2 preve lote de 1.000 imagens do
conjunto de teste, medicao executada por ``automacao/run_experiments.py`` sobre o
corpus real. Aqui verifica-se que a instrumentacao de tempo funciona e que a
ordem de grandeza esta muito abaixo do limite.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from codigo.configuracao import IMAGE_SIZE                        # noqa: E402
from codigo.tecnicas.t03_benford import T03Benford        # noqa: E402
from codigo.tecnicas.t05_fusao import T05Fusion, build_score_matrix   # noqa: E402

# RNF01: tempo total de analise inferior a 30 segundos por imagem.
LIMITE_SEGUNDOS = 30.0


@pytest.fixture
def batch(tmp_path) -> tuple[list[Path], np.ndarray]:
    rng = np.random.default_rng(42)
    paths, labels = [], []
    for index in range(12):
        array = rng.integers(0, 256, size=(IMAGE_SIZE, IMAGE_SIZE, 3), dtype=np.uint8)
        path = tmp_path / f"imagem_{index:03d}.png"
        Image.fromarray(array).save(path)
        paths.append(path)
        labels.append(index % 2)
    return paths, np.array(labels)


def test_t03_dentro_do_limite(batch):
    paths, labels = batch
    technique = T03Benford().fit(paths, labels)

    result = technique.run(paths)
    segundos_por_imagem = result.ms_per_image / 1000.0

    assert np.isfinite(segundos_por_imagem)
    assert segundos_por_imagem < LIMITE_SEGUNDOS, (
        f"T03 levou {segundos_por_imagem:.2f} s/imagem, acima do limite de "
        f"{LIMITE_SEGUNDOS} s (RNF01)"
    )


def test_t05_custo_desprezivel(batch):
    """A fusao opera sobre quatro valores; seu custo proprio e residual."""
    _, labels = batch
    rng = np.random.default_rng(0)
    scores = {t: rng.uniform(size=labels.size) for t in ("T01", "T02", "T03", "T04")}
    matrix = build_score_matrix(scores, labels.size)

    fusion = T05Fusion().fit_scores(matrix, labels)

    import time
    started = time.perf_counter()
    fusion.predict_proba_scores(matrix)
    elapsed = time.perf_counter() - started

    assert elapsed / labels.size < 0.1


def test_instrumentacao_de_tempo_reporta_valor_finito(batch):
    """Regressao: o tempo de inferencia deve chegar as metricas, nao ficar NaN."""
    paths, labels = batch
    technique = T03Benford().fit(paths, labels)

    result = technique.run(paths)
    assert result.elapsed_seconds > 0
    assert np.isfinite(result.ms_per_image)
    assert result.n_samples == len(paths)

    metrics = technique.score(paths, labels)
    assert "ms_per_image" in metrics
    assert np.isfinite(metrics["ms_per_image"])


def test_ms_por_imagem_com_conjunto_vazio():
    from codigo.tecnicas.base import TechniqueResult

    result = TechniqueResult("T00", np.empty(0), 0.0, 0)
    assert np.isnan(result.ms_per_image)
