"""Gera um corpus sintetico reduzido para validar o pipeline.

O dataset de Corvi et al. (2024) tem 360.000 imagens e exige download longo.
Este script produz um corpus com a MESMA ESTRUTURA de diretorios e um sinal
detectavel plantado, permitindo exercitar todo o fluxo -- verificacao de
integridade, particionamento, treinamento de T01/T03, fusao em T05, os tres
protocolos e a interface -- antes de dispor do corpus real.

As imagens "reais" sao texturas com espectro do tipo 1/f, caracteristico de
cenas naturais. As "sinteticas" recebem dois artefatos tipicos de modelos
generativos: um padrao periodico de upsampling (pico espectral em alta
frequencia) e uma distribuicao de intensidades mais suave.

ATENCAO: este corpus serve exclusivamente a verificacao funcional do codigo.
As metricas obtidas sobre ele NAO tem valor cientifico e nao devem ser
reportadas na monografia.

Uso::

    python scripts/make_sample_corpus.py --output data/amostra --per-class 200
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import ALL_GENERATORS, IMAGE_SIZE, REAL_SOURCES   # noqa: E402


def pink_noise_image(rng: np.random.Generator, size: int) -> np.ndarray:
    """Textura com espectro de potencia proporcional a 1/f.

    Imagens naturais exibem esse decaimento espectral, o que torna a textura
    um substituto razoavel para fins de teste funcional.
    """
    frequencies_y = np.fft.fftfreq(size)[:, None]
    frequencies_x = np.fft.fftfreq(size)[None, :]
    radius = np.sqrt(frequencies_y**2 + frequencies_x**2)
    radius[0, 0] = 1.0                       # evita divisao por zero na componente DC

    channels = []
    for _ in range(3):
        white = rng.normal(size=(size, size))
        filtered = np.fft.ifft2(np.fft.fft2(white) / (radius ** 1.0)).real
        filtered -= filtered.min()
        peak = filtered.max()
        channels.append(filtered / peak if peak > 0 else filtered)
    array = np.stack(channels, axis=2)
    return np.clip(array * 255, 0, 255).astype(np.uint8)


def synthetic_artifact_image(rng: np.random.Generator, size: int, strength: float) -> np.ndarray:
    """Textura com artefatos periodicos de upsampling.

    Gera a imagem em metade da resolucao e a amplia por vizinho mais proximo,
    reproduzindo o padrao de blocos 2 x 2 que introduz picos regulares no
    espectro -- o mesmo tipo de evidencia que T02 explora.
    """
    half = pink_noise_image(rng, size // 2)
    upsampled = np.repeat(np.repeat(half, 2, axis=0), 2, axis=1)

    # Modulacao periodica adicional, analoga ao checkerboard de convolucoes
    # transpostas.
    grid_y, grid_x = np.meshgrid(np.arange(size), np.arange(size), indexing="ij")
    ripple = np.sin(np.pi * grid_y / 2) * np.sin(np.pi * grid_x / 2)
    modulated = upsampled + strength * 12.0 * ripple[:, :, None]

    # Suavizacao leve das intensidades, reproduzindo a menor variancia local
    # observada em saidas de modelos generativos.
    smoothed = 0.85 * modulated + 0.15 * modulated.mean()
    return np.clip(smoothed, 0, 255).astype(np.uint8)


def main() -> int:
    parser = argparse.ArgumentParser(description="Gera corpus sintetico para teste funcional")
    parser.add_argument("--output", type=Path, default=Path("data/amostra"))
    parser.add_argument("--per-class", type=int, default=200,
                        help="numero de imagens por classe (dividido entre fontes/geradores)")
    parser.add_argument("--size", type=int, default=IMAGE_SIZE)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--strength", type=float, default=1.0,
                        help="intensidade do artefato sintetico; menor valor torna a tarefa mais dificil")
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    output = args.output

    per_source = max(1, args.per_class // len(REAL_SOURCES))
    print(f"Gerando {per_source * len(REAL_SOURCES)} imagens reais ...")
    for source in REAL_SOURCES:
        directory = output / "real" / source
        directory.mkdir(parents=True, exist_ok=True)
        for index in range(per_source):
            array = pink_noise_image(rng, args.size)
            Image.fromarray(array).save(directory / f"{source}_{index:05d}.png")

    per_generator = max(1, args.per_class // len(ALL_GENERATORS))
    print(f"Gerando {per_generator * len(ALL_GENERATORS)} imagens sinteticas ...")
    for generator in ALL_GENERATORS:
        directory = output / "fake" / generator
        directory.mkdir(parents=True, exist_ok=True)
        for index in range(per_generator):
            array = synthetic_artifact_image(rng, args.size, args.strength)
            Image.fromarray(array).save(directory / f"{generator}_{index:05d}.png")

    print(f"\nCorpus de amostra gravado em {output}")
    print("\nProximos passos:")
    print(f"  python scripts/prepare_dataset.py --corpus {output}")
    print("  python scripts/run_experiments.py --protocol standard --techniques T01 T03 T05 --fit")
    print("\nLembrete: metricas obtidas sobre este corpus nao devem ser reportadas na monografia.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
