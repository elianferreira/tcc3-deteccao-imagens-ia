"""Extrai o dataset de mascaras objeto-sombra no layout esperado pelo codigo oficial.

O arquivo publicado traz tudo sob um diretorio ``<Nome>_OS/``; o codigo oficial
espera os diretorios ``<Nome>_shadow/`` e ``<Nome>_object/`` diretamente em
``dataset/``. O README oficial resolve isso com ``mv <Nome>_OS/* ./``; aqui a
remocao do nivel extra e feita durante a extracao, evitando escrever 500 mil
arquivos duas vezes.

Uso::

    python scripts/extrair_os_dataset.py --zip data/downloads/Kandinsky_Outdoor_OS.zip
"""

from __future__ import annotations

import argparse
import sys
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import DATA_DIR, EXTERNAL      # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Extrai o dataset de mascaras objeto-sombra")
    parser.add_argument("--zip", type=Path,
                        default=DATA_DIR / "downloads" / "Kandinsky_Outdoor_OS.zip")
    parser.add_argument("--output", type=Path, default=EXTERNAL.geometry_repo / "dataset")
    args = parser.parse_args()

    if not args.zip.exists():
        print(f"ERRO: arquivo nao encontrado em {args.zip}")
        return 1

    args.output.mkdir(parents=True, exist_ok=True)
    print(f"Extraindo {args.zip.name} -> {args.output}")

    extraidos = 0
    pulados = 0
    with zipfile.ZipFile(args.zip) as arquivo:
        for info in arquivo.infolist():
            if info.is_dir():
                continue
            partes = info.filename.split("/")
            if len(partes) < 2:
                continue
            # Remove o nivel "<Nome>_OS/" do inicio do caminho.
            destino = args.output.joinpath(*partes[1:])
            if destino.exists() and destino.stat().st_size == info.file_size:
                pulados += 1
                continue
            destino.parent.mkdir(parents=True, exist_ok=True)
            with arquivo.open(info) as origem, destino.open("wb") as saida:
                saida.write(origem.read())
            extraidos += 1
            if extraidos % 20_000 == 0:
                print(f"  {extraidos} arquivos extraidos ...", flush=True)

    print(f"\nExtraidos: {extraidos}")
    print(f"Ja presentes: {pulados}")
    for sub in sorted(p for p in args.output.iterdir() if p.is_dir()):
        total = sum(1 for _ in sub.rglob("*") if _.is_file())
        print(f"  {sub.name:<32} {total:>7} arquivos")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
