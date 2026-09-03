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
analise dos resultados. Ver documentacao/DECISOES_METODOLOGICAS.md, secao 6.

Onde isto e usado
-----------------
Duas vezes, e a igualdade entre as duas e o que faz a tela corresponder
ao capitulo:

    automacao/normalize_corpus.py    normaliza o corpus inteiro, em lote
    codigo/preprocessamento/         normaliza cada upload da interface
        envio_interface.py
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path

from PIL import Image

from ..configuracao import IMAGE_SIZE


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
