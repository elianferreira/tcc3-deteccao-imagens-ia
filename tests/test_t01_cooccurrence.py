"""Testes unitarios de T01 - coocorrencia RGB + CNN (Secao 3.6.2).

Isolados em arquivo proprio porque exigem PyTorch, que e instalado apenas no
ambiente principal; os testes de T03 e das metricas permanecem executaveis sem
essa dependencia.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

torch = pytest.importorskip("torch", reason="T01 requer PyTorch")

from src.config import IMAGE_SIZE  # noqa: E402


@pytest.fixture
def sample_image() -> Image.Image:
    rng = np.random.default_rng(42)
    array = rng.integers(0, 256, size=(IMAGE_SIZE, IMAGE_SIZE, 3), dtype=np.uint8)
    return Image.fromarray(array, mode="RGB")

def test_coocorrencia_dimensoes_e_normalizacao(sample_image):
    from src.techniques.t01_cooccurrence import rgb_cooccurrence_tensor

    tensor = rgb_cooccurrence_tensor(sample_image)
    assert tensor.shape == (3, 256, 256)
    assert np.isfinite(tensor).all()
    assert (tensor >= 0).all()
    # Cada canal e normalizado para somar 1.
    for channel in range(3):
        assert np.isclose(tensor[channel].sum(), 1.0, atol=1e-5)


def test_coocorrencia_padrao_conhecido():
    """Imagem de intensidade constante concentra toda a massa em (v, v)."""
    from src.techniques.t01_cooccurrence import cooccurrence_matrix

    channel = np.full((16, 16), 77, dtype=np.uint8)
    matrix = cooccurrence_matrix(channel, offset=(0, 1))
    assert np.isclose(matrix[77, 77], 1.0)
    assert np.isclose(matrix.sum(), 1.0)


def test_coocorrencia_gradiente_horizontal():
    """Em um gradiente de passo unitario, a massa fica na diagonal (i, i+1)."""
    from src.techniques.t01_cooccurrence import cooccurrence_matrix

    channel = np.tile(np.arange(16, dtype=np.uint8), (16, 1))
    matrix = cooccurrence_matrix(channel, offset=(0, 1))
    for i in range(15):
        assert matrix[i, i + 1] > 0
    assert np.isclose(np.trace(matrix), 0.0)


def test_coocorrencia_reprodutivel(sample_image):
    from src.techniques.t01_cooccurrence import rgb_cooccurrence_tensor

    assert np.array_equal(rgb_cooccurrence_tensor(sample_image),
                          rgb_cooccurrence_tensor(sample_image))


def test_arquitetura_t01_saida_escalar():
    from src.techniques.t01_cooccurrence import CooccurrenceCNN

    model = CooccurrenceCNN()
    output = model(torch.randn(4, 3, 256, 256))
    assert output.shape == (4,)
    assert torch.isfinite(output).all()


def test_arquitetura_t01_contagem_de_filtros():
    """Confere os 32, 32, 64, 64, 128 e 128 filtros do trabalho original."""
    from src.techniques.t01_cooccurrence import CooccurrenceCNN

    model = CooccurrenceCNN()
    conv_layers = [m for m in model.features if isinstance(m, torch.nn.Conv2d)]
    assert [c.out_channels for c in conv_layers] == [32, 32, 64, 64, 128, 128]
    assert [c.kernel_size[0] for c in conv_layers] == [3, 5, 3, 5, 3, 5]
    assert sum(1 for m in model.features if isinstance(m, torch.nn.MaxPool2d)) == 3
