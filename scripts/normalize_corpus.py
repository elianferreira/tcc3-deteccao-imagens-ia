"""Normalizacao do corpus: resolucao e formato uniformes.

Motivacao
---------
A verificacao de integridade (Etapa 1) sinalizou 16.638 de 22.638 imagens com
resolucao diferente de 256 x 256. O corpus reunido a partir das fontes publicas
mistura escalas e formatos muito distintos:

    real/coco                ~640 x 427   JPEG
    fake/latent_diffusion     256 x 256   PNG
    fake/glide                256 x 256   PNG
    fake/flux                1024 x 768   PNG
    fake/midjourney_v6_1     1024 x 1024  PNG
    fake/adobe_firefly       2304 x 1792  PNG

Isso introduz DOIS confundidores sobrepostos entre as classes -- tamanho e
formato de compressao --, qualquer um deles suficiente para um detector separar
real de sintetico sem aprender evidencia forense alguma. E o mesmo vies de
conjunto de dados discutido na introducao do TCC 2.

Ha ainda um efeito pratico: as imagens de 2304 x 1792 tem 63 vezes mais pixels
que 256 x 256, e o SPAI (T02) opera em resolucao nativa por construcao. Isso
inviabilizou a inferencia na GPU de 6 GB disponivel.

O que este script faz
---------------------
1. Redimensiona preservando a proporcao: o menor lado vai a 256 e o excedente
   do maior lado e removido por recorte centralizado. Isso evita a distorcao
   geometrica que um resize direto para 256 x 256 produziria -- distorcao que
   alteraria justamente as relacoes de perspectiva e as estatisticas locais que
   as tecnicas analisam.
2. Grava tudo em PNG, eliminando a segunda compressao JPEG que distinguiria as
   imagens reais das sinteticas.

Limitacao declarada
-------------------
O SPAI foi projetado para resolucao arbitraria -- Spectral Context Attention e
uma de suas contribuicoes. Avalia-lo apenas em 256 x 256 subestima o metodo.
Essa restricao e consequencia do hardware disponivel e deve ser declarada na
analise dos resultados. Ver docs/DECISOES_METODOLOGICAS.md, secao 6.

Uso::

    python scripts/normalize_corpus.py --input data/corvi2024 --output data/corvi2024_norm
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import DATA_DIR, IMAGE_SIZE      # noqa: E402

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}


def resize_and_center_crop(image: Image.Image, size: int = IMAGE_SIZE) -> Image.Image:
    """Menor lado a ``size``, depois recorte centralizado ``size`` x ``size``.

    A ordem importa. Redimensionar diretamente para um quadrado distorceria a
    geometria de imagens nao quadradas; reduzir primeiro pelo menor lado e
    recortar preserva as proporcoes originais da cena.
    """
    image = image.convert("RGB")
    width, height = image.size

    if (width, height) == (size, size):
        return image

    scale = size / min(width, height)
    new_size = (max(size, round(width * scale)), max(size, round(height * scale)))
    # LANCZOS preserva melhor o conteudo de alta frequencia na reducao, que e
    # justamente onde residem os artefatos que T01 e T03 exploram.
    image = image.resize(new_size, Image.LANCZOS)

    left = (image.width - size) // 2
    top = (image.height - size) // 2
    return image.crop((left, top, left + size, top + size))


def normalize_tree(input_root: Path, output_root: Path, size: int) -> dict:
    report: dict = {
        "processadas": 0,
        "ja_conformes": 0,
        "falhas": [],
        "tamanhos_originais": Counter(),
        "por_grupo": Counter(),
    }

    for source in sorted(input_root.rglob("*")):
        if not source.is_file() or source.suffix.lower() not in IMAGE_EXTENSIONS:
            continue

        relative = source.relative_to(input_root)
        # Formato unico: PNG, sem segunda compressao com perdas.
        destination = (output_root / relative).with_suffix(".png")
        destination.parent.mkdir(parents=True, exist_ok=True)

        if destination.exists():
            report["processadas"] += 1
            continue

        try:
            with Image.open(source) as image:
                original = image.size
                if original == (size, size) and source.suffix.lower() == ".png":
                    report["ja_conformes"] += 1
                normalized = resize_and_center_crop(image, size)
                normalized.save(destination, format="PNG")
        except Exception as error:                      # noqa: BLE001
            report["falhas"].append({"path": str(source), "erro": str(error)})
            continue

        report["tamanhos_originais"][f"{original[0]}x{original[1]}"] += 1
        report["por_grupo"][str(relative.parent)] += 1
        report["processadas"] += 1

        if report["processadas"] % 2000 == 0:
            print(f"  {report['processadas']} imagens normalizadas ...")

    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Normaliza resolucao e formato do corpus")
    parser.add_argument("--input", type=Path, default=DATA_DIR / "corvi2024")
    parser.add_argument("--output", type=Path, default=DATA_DIR / "corvi2024_norm")
    parser.add_argument("--size", type=int, default=IMAGE_SIZE)
    args = parser.parse_args()

    if not args.input.exists():
        print(f"ERRO: corpus nao encontrado em {args.input}")
        return 1

    print(f"Normalizando {args.input} -> {args.output}")
    print(f"  resolucao alvo: {args.size} x {args.size} (menor lado + recorte central)")
    print(f"  formato alvo:   PNG\n")

    report = normalize_tree(args.input, args.output, args.size)

    print("\n" + "=" * 68)
    print(f"Imagens normalizadas ......... {report['processadas']}")
    print(f"  ja conformes ............... {report['ja_conformes']}")
    print(f"  falhas ..................... {len(report['falhas'])}")
    print("-" * 68)
    print("Resolucoes originais mais frequentes:")
    for tamanho, contagem in report["tamanhos_originais"].most_common(8):
        print(f"  {tamanho:<14} {contagem:>6}")
    print("=" * 68)

    if report["falhas"]:
        print(f"\nATENCAO: {len(report['falhas'])} arquivos falharam:")
        for falha in report["falhas"][:5]:
            print(f"  {falha['path']}: {falha['erro']}")

    serializable = {
        k: (dict(v) if isinstance(v, Counter) else v) for k, v in report.items()
    }
    info_path = args.output / "normalizacao_info.json"
    info_path.parent.mkdir(parents=True, exist_ok=True)
    info_path.write_text(json.dumps(serializable, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nRelatorio gravado em {info_path}")

    print("\nProximos passos:")
    print(f"  python scripts/prepare_dataset.py --corpus {args.output} --protocol standard")
    print(f"  python scripts/prepare_dataset.py --corpus {args.output} --protocol ood")
    return 0 if not report["falhas"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
