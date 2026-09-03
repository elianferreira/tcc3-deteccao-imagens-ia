"""Perturbacoes do protocolo de robustez (Secao 3.6.1).

O conjunto de teste e submetido a perturbacoes progressivas -- compressao JPEG
com Q = 50/70/85, ruido Gaussiano com sigma = 1/3/5 e redimensionamento para
50%/70%/85% -- e as metricas sao recalculadas. Avalia a estabilidade dos
detectores diante de degradacoes comuns em redes sociais.

As imagens perturbadas sao materializadas em disco porque T02 e T04 sao
executadas por subprocesso sobre os pipelines oficiais e recebem caminhos de
arquivo, nao arrays em memoria.
"""

from __future__ import annotations

import io
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterator, Sequence

import numpy as np
from PIL import Image

from ..configuracao import GAUSSIAN_SIGMAS, JPEG_QUALITIES, RESIZE_FACTORS


@dataclass(frozen=True)
class Perturbation:
    name: str
    apply: Callable[[Image.Image], Image.Image]
    #: Extensao de gravacao; JPEG precisa preservar o artefato de compressao.
    extension: str = ".png"


# ---------------------------------------------------------------------------
# Operadores
# ---------------------------------------------------------------------------


def jpeg_compression(quality: int) -> Callable[[Image.Image], Image.Image]:
    """Recompressao JPEG com o fator de qualidade indicado."""
    def operator(image: Image.Image) -> Image.Image:
        buffer = io.BytesIO()
        image.convert("RGB").save(buffer, format="JPEG", quality=int(quality))
        buffer.seek(0)
        with Image.open(buffer) as compressed:
            return compressed.convert("RGB").copy()
    return operator


def gaussian_noise(sigma: float, seed: int = 42) -> Callable[[Image.Image], Image.Image]:
    """Ruido Gaussiano aditivo com desvio padrao em niveis de intensidade.

    A semente e derivada do conteudo da imagem para que a mesma imagem receba
    sempre o mesmo ruido, preservando a reprodutibilidade exigida por RNF02 sem
    tornar identico o ruido aplicado a imagens distintas.
    """
    def operator(image: Image.Image) -> Image.Image:
        array = np.asarray(image.convert("RGB"), dtype=np.float64)
        content_seed = (seed + int(array.sum()) % (2**31)) % (2**32)
        rng = np.random.default_rng(content_seed)
        noisy = array + rng.normal(0.0, float(sigma), size=array.shape)
        return Image.fromarray(np.clip(noisy, 0, 255).astype(np.uint8), mode="RGB")
    return operator


def resize(factor: float) -> Callable[[Image.Image], Image.Image]:
    """Redimensiona para ``factor`` da resolucao original.

    A imagem permanece na resolucao reduzida: retornar ao tamanho original
    introduziria artefatos de interpolacao adicionais que nao fazem parte da
    degradacao avaliada. T02 processa imagens em qualquer resolucao por
    construcao (Spectral Context Attention); as demais tecnicas reajustam a
    entrada no proprio pre-processamento.
    """
    def operator(image: Image.Image) -> Image.Image:
        width, height = image.size
        new_size = (max(1, int(round(width * factor))), max(1, int(round(height * factor))))
        return image.convert("RGB").resize(new_size, Image.BICUBIC)
    return operator


def identity() -> Callable[[Image.Image], Image.Image]:
    def operator(image: Image.Image) -> Image.Image:
        return image.convert("RGB").copy()
    return operator


# ---------------------------------------------------------------------------
# Catalogo
# ---------------------------------------------------------------------------


def build_perturbations() -> list[Perturbation]:
    """Grade completa do protocolo de robustez, incluindo a condicao limpa."""
    catalog = [Perturbation("clean", identity(), ".png")]
    catalog += [
        Perturbation(f"jpeg_q{q}", jpeg_compression(q), ".jpg") for q in JPEG_QUALITIES
    ]
    catalog += [
        Perturbation(f"noise_sigma{sigma:g}", gaussian_noise(sigma), ".png")
        for sigma in GAUSSIAN_SIGMAS
    ]
    catalog += [
        Perturbation(f"resize_{int(f * 100)}", resize(f), ".png") for f in RESIZE_FACTORS
    ]
    return catalog


PERTURBATIONS = {p.name: p for p in build_perturbations()}


def materialize(
    paths: Sequence[Path],
    perturbation: Perturbation,
    output_dir: Path,
) -> list[Path]:
    """Aplica a perturbacao e grava as imagens resultantes.

    Os nomes de arquivo recebem um prefixo posicional para evitar colisoes
    entre imagens homonimas provenientes de geradores distintos.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    written: list[Path] = []
    for index, source in enumerate(paths):
        destination = output_dir / f"{index:07d}_{Path(source).stem}{perturbation.extension}"
        with Image.open(source) as image:
            result = perturbation.apply(image)
            if perturbation.extension in (".jpg", ".jpeg"):
                # A perturbacao ja produziu o artefato JPEG desejado; grava-se
                # em qualidade maxima para nao acrescentar uma segunda
                # compressao sobre a primeira.
                result.save(destination, format="JPEG", quality=100, subsampling=0)
            else:
                result.save(destination, format="PNG")
        written.append(destination)
    return written


def iter_perturbations(names: Sequence[str] | None = None) -> Iterator[Perturbation]:
    if names is None:
        yield from build_perturbations()
        return
    for name in names:
        if name not in PERTURBATIONS:
            raise KeyError(f"perturbacao desconhecida: {name}; disponiveis: {list(PERTURBATIONS)}")
        yield PERTURBATIONS[name]
