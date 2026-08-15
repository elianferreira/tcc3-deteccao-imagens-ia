"""Testes unitarios dos modulos de extracao de caracteristicas (Secao 3.6.2).

Verificam: (i) dimensionalidade da saida esperada; (ii) ausencia de valores NaN
ou infinitos; e (iii) reprodutibilidade -- a mesma imagem de entrada deve
produzir saida numericamente identica em duas execucoes com semente fixada.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import IMAGE_SIZE, T03            # noqa: E402
from src.techniques.t03_benford import (          # noqa: E402
    ZIGZAG_ORDER, benford_features, benford_reference, first_digit,
    first_digit_distribution, jensen_shannon_divergence,
    quantized_dct_coefficients, renyi_divergence, tsallis_divergence,
)


@pytest.fixture
def sample_image() -> Image.Image:
    rng = np.random.default_rng(42)
    array = rng.integers(0, 256, size=(IMAGE_SIZE, IMAGE_SIZE, 3), dtype=np.uint8)
    return Image.fromarray(array, mode="RGB")


# ---------------------------------------------------------------------------
# Lei de Benford
# ---------------------------------------------------------------------------


def test_benford_reference_soma_um():
    for base in T03.bases:
        reference = benford_reference(base)
        assert reference.shape == (base - 1,)
        assert np.isclose(reference.sum(), 1.0)
        # A lei e monotonicamente decrescente no primeiro digito.
        assert np.all(np.diff(reference) < 0)


def test_benford_base_10_valores_conhecidos():
    reference = benford_reference(10)
    assert np.isclose(reference[0], 0.30103, atol=1e-5)   # P(1)
    assert np.isclose(reference[8], 0.04576, atol=1e-5)   # P(9)


def test_first_digit_casos_conhecidos():
    values = np.array([1.0, 23.0, -456.0, 0.0071, 9999.0])
    assert np.array_equal(first_digit(values, 10), np.array([1, 2, 4, 7, 9]))


def test_first_digit_rejeita_zeros():
    with pytest.raises(ValueError):
        first_digit(np.array([0.0, 1.0]), 10)


def test_first_digit_base_20():
    # Na base 20, o valor 25 e escrito como (1)(5): primeiro digito 1.
    assert first_digit(np.array([25.0]), 20)[0] == 1
    assert first_digit(np.array([19.0]), 20)[0] == 19


def test_distribuicao_sem_coeficientes_significativos():
    distribution = first_digit_distribution(np.zeros(100), 10)
    assert np.allclose(distribution, 1.0 / 9.0)


# ---------------------------------------------------------------------------
# Divergencias
# ---------------------------------------------------------------------------


def test_divergencias_nulas_para_distribuicoes_identicas():
    reference = benford_reference(10)
    assert np.isclose(jensen_shannon_divergence(reference, reference), 0.0, atol=1e-10)
    assert np.isclose(renyi_divergence(reference, reference, 0.5), 0.0, atol=1e-10)
    assert np.isclose(tsallis_divergence(reference, reference, 0.5), 0.0, atol=1e-10)


def test_jensen_shannon_simetrica_e_limitada():
    p = benford_reference(10)
    q = np.full(9, 1.0 / 9.0)
    value = jensen_shannon_divergence(p, q)
    assert np.isclose(value, jensen_shannon_divergence(q, p))
    assert 0.0 <= value <= 1.0


def test_divergencias_positivas_para_distribuicoes_distintas():
    p = benford_reference(10)
    q = np.full(9, 1.0 / 9.0)
    assert jensen_shannon_divergence(p, q) > 0
    assert renyi_divergence(p, q, 0.5) > 0
    assert tsallis_divergence(p, q, 0.5) > 0


def test_renyi_rejeita_alpha_invalido():
    p = benford_reference(10)
    with pytest.raises(ValueError):
        renyi_divergence(p, p, 1.0)


# ---------------------------------------------------------------------------
# DCT e vetor de caracteristicas
# ---------------------------------------------------------------------------


def test_zigzag_e_permutacao_valida():
    assert ZIGZAG_ORDER.shape == (64,)
    assert sorted(ZIGZAG_ORDER.tolist()) == list(range(64))
    assert ZIGZAG_ORDER[0] == 0        # componente DC
    assert ZIGZAG_ORDER[1] == 1        # primeira AC horizontal


def test_dct_quantizada_dimensoes(sample_image):
    coefficients, table = quantized_dct_coefficients(sample_image, 90)
    n_blocks = (IMAGE_SIZE // 8) ** 2
    assert coefficients.shape == (n_blocks, 64)
    assert table.shape == (8, 8)
    assert np.isfinite(coefficients).all()


def test_dct_maior_qualidade_menor_quantizacao(sample_image):
    _, table_low = quantized_dct_coefficients(sample_image, 80)
    _, table_high = quantized_dct_coefficients(sample_image, 100)
    # Qualidade maior implica passos de quantizacao menores.
    assert table_high.mean() < table_low.mean()


def test_vetor_benford_tem_540_dimensoes(sample_image):
    features = benford_features(sample_image)
    assert features.shape == (540,)
    assert features.shape == (T03.n_features,)


def test_vetor_benford_sem_nan_ou_infinito(sample_image):
    features = benford_features(sample_image)
    assert np.isfinite(features).all()


def test_vetor_benford_reprodutivel(sample_image):
    """A mesma imagem deve produzir saida numericamente identica (RNF02)."""
    assert np.array_equal(benford_features(sample_image), benford_features(sample_image))


def test_vetor_benford_discrimina_imagens_distintas(sample_image):
    uniform = Image.fromarray(np.full((IMAGE_SIZE, IMAGE_SIZE, 3), 128, dtype=np.uint8))
    assert not np.allclose(benford_features(sample_image), benford_features(uniform))
