"""Testes do agregador de mascaras de T04 (Secao 3.6.2).

O agregador reduz as ``N`` mascaras de instancia do SSISv2 aos dois mapas
binarios que o classificador oficial consome. Ele e a unica peca nova entre o
extrator e o classificador ja validado, e um erro seu -- classes trocadas, uniao
mal feita, nivel de cinza indevido -- nao produziria excecao, apenas metricas
silenciosamente erradas. Dai a cobertura.

Rodam em CPU e sem detectron2: o agregador e uma funcao pura sobre arrays, e o
modulo so importa torch, cv2 e PIL dentro das funcoes que precisam deles.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ))

# Importado por caminho porque scripts/wsl nao e um pacote e o modulo so e
# executavel dentro do WSL; aqui interessa apenas a funcao pura.
_ESPEC = importlib.util.spec_from_file_location(
    "extrair_object_shadow_ssis",
    RAIZ / "scripts" / "wsl" / "extrair_object_shadow_ssis.py",
)
_MODULO = importlib.util.module_from_spec(_ESPEC)
_ESPEC.loader.exec_module(_MODULO)

agregar_por_classe = _MODULO.agregar_por_classe
caminho_windows_para_wsl = _MODULO.caminho_windows_para_wsl
CLASSE_OBJETO = _MODULO.CLASSE_OBJETO
CLASSE_SOMBRA = _MODULO.CLASSE_SOMBRA


def _mascara(altura=8, largura=8, regiao=None) -> np.ndarray:
    """Mascara float32 como o SSISv2 a devolve, com 1,0 dentro da regiao."""
    m = np.zeros((altura, largura), dtype=np.float32)
    if regiao is not None:
        (y0, y1), (x0, x1) = regiao
        m[y0:y1, x0:x1] = 1.0
    return m


def test_separa_objeto_de_sombra_pela_classe():
    """Classe 0 vai para o mapa de objeto, classe 1 para o de sombra."""
    obj = _mascara(regiao=((0, 4), (0, 4)))
    som = _mascara(regiao=((4, 8), (4, 8)))
    mascaras = np.stack([obj, som])
    classes = np.array([CLASSE_OBJETO, CLASSE_SOMBRA])

    objeto, sombra = agregar_por_classe(mascaras, classes)

    assert objeto[0, 0] == 255 and objeto[7, 7] == 0
    assert sombra[7, 7] == 255 and sombra[0, 0] == 0


def test_uniao_de_multiplas_instancias_da_mesma_classe():
    """Duas instancias de objeto ocupam o mesmo mapa, sem se anularem."""
    a = _mascara(regiao=((0, 2), (0, 2)))
    b = _mascara(regiao=((6, 8), (6, 8)))
    mascaras = np.stack([a, b])
    classes = np.array([CLASSE_OBJETO, CLASSE_OBJETO])

    objeto, sombra = agregar_por_classe(mascaras, classes)

    assert objeto[0, 0] == 255
    assert objeto[7, 7] == 255
    assert objeto.sum() == 255 * 8          # 4 pixels de cada instancia
    assert sombra.max() == 0                # nenhuma instancia de sombra


def test_instancias_sobrepostas_nao_somam():
    """A uniao e logica: sobreposicao continua valendo 255, nao 510."""
    a = _mascara(regiao=((0, 4), (0, 4)))
    b = _mascara(regiao=((2, 6), (2, 6)))
    objeto, _ = agregar_por_classe(np.stack([a, b]), np.array([0, 0]))

    assert objeto.max() == 255
    assert objeto.dtype == np.uint8


def test_saida_e_estritamente_binaria():
    """Nenhum nivel intermediario: as mascaras oficiais nao os tem.

    Verificado contra amostra de 200 arquivos de cada classe do conjunto dos
    autores, onde a massa no intervalo [20, 235] e exatamente zero.
    """
    suave = np.zeros((8, 8), dtype=np.float32)
    suave[0, 0] = 0.9
    suave[0, 1] = 0.51
    suave[1, 0] = 0.49       # abaixo do limiar
    suave[1, 1] = 0.05

    objeto, _ = agregar_por_classe(suave[None, ...], np.array([CLASSE_OBJETO]))

    assert set(np.unique(objeto)).issubset({0, 255})
    assert objeto[0, 0] == 255 and objeto[0, 1] == 255
    assert objeto[1, 0] == 0 and objeto[1, 1] == 0


def test_mascaras_booleanas_tambem_sao_aceitas():
    """O limiar de 0,5 cobre bool e float sem ramo especial."""
    booleana = np.zeros((4, 4), dtype=bool)
    booleana[0, 0] = True
    objeto, _ = agregar_por_classe(booleana[None, ...], np.array([CLASSE_OBJETO]))
    assert objeto[0, 0] == 255 and objeto.sum() == 255


def test_imagem_sem_deteccao_produz_par_preto():
    """Caso legitimo e frequente; nao deve levantar excecao.

    A forma (0, H, W) preserva a resolucao, entao os mapas saem pretos no
    tamanho certo em vez de degenerados.
    """
    vazio = np.zeros((0, 8, 8), dtype=np.float32)
    objeto, sombra = agregar_por_classe(vazio, np.array([], dtype=int))

    assert objeto.shape == (8, 8) and sombra.shape == (8, 8)
    assert objeto.max() == 0 and sombra.max() == 0
    assert objeto.dtype == np.uint8


def test_apenas_sombra_detectada():
    """Sombra sem objeto: o mapa de objeto sai preto, nao ausente."""
    som = _mascara(regiao=((0, 4), (0, 4)))
    objeto, sombra = agregar_por_classe(som[None, ...], np.array([CLASSE_SOMBRA]))

    assert sombra.max() == 255
    assert objeto.max() == 0
    assert objeto.shape == sombra.shape


def test_rejeita_contagens_incompativeis():
    mascaras = np.zeros((2, 4, 4), dtype=np.float32)
    with pytest.raises(ValueError, match="instancias"):
        agregar_por_classe(mascaras, np.array([0]))


def test_rejeita_forma_invalida():
    with pytest.raises(ValueError, match="pred_masks"):
        agregar_por_classe(np.zeros((4, 4), dtype=np.float32), np.array([0]))


@pytest.mark.parametrize("bruto,esperado", [
    (r"C:\Users\ferre\projects\x\a.png", "/mnt/c/Users/ferre/projects/x/a.png"),
    (r"D:\dados\b.png", "/mnt/d/dados/b.png"),
    ("/mnt/c/ja/posix.png", "/mnt/c/ja/posix.png"),
])
def test_conversao_de_caminho_windows(bruto, esperado):
    assert caminho_windows_para_wsl(bruto).as_posix() == esperado
