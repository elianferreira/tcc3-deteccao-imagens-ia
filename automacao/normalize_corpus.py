"""Linha de comando da normalizacao do corpus.

A logica -- ``resize_and_center_crop`` e ``normalize_tree`` -- vive em
``codigo/preprocessamento/normalizacao.py``, junto do restante do que se
aplica a imagem antes das tecnicas. E o MESMO codigo que a interface usa
em cada upload, e essa igualdade e o que faz a tela corresponder ao
Capitulo 4. Duplicar aqui seria abrir espaco para as duas divergirem.

Uso::

    python automacao/normalize_corpus.py --input dados/corvi2024 \n                                         --output dados/corvi2024_norm
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from codigo.configuracao import DATA_DIR, IMAGE_SIZE      # noqa: E402
from codigo.preprocessamento.normalizacao import (        # noqa: E402
    IMAGE_EXTENSIONS, normalize_tree, resize_and_center_crop,
)

__all__ = ["IMAGE_EXTENSIONS", "normalize_tree", "resize_and_center_crop", "main"]


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
    print(f"  python automacao/prepare_dataset.py --corpus {args.output} --protocol standard")
    print(f"  python automacao/prepare_dataset.py --corpus {args.output} --protocol ood")
    return 0 if not report["falhas"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
