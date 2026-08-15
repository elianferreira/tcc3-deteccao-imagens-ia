"""Etapa 1 - Organizacao do corpus no layout real/ e fake/.

Extrai os arquivos baixados por ``download_dataset.py`` e os reorganiza na
estrutura que ``prepare_dataset.py`` espera, normalizando os nomes dos
geradores para os identificadores da Secao 3.5.1.

Mapeamento dos 13 geradores do benchmark
----------------------------------------
Synthbuster (9): glide, stable-diffusion-1-3, stable-diffusion-1-4,
stable-diffusion-2, stable-diffusion-xl, dalle2, dalle3, firefly, midjourney-v5
Extra generators (4): flux, gigagan, midjourney-v6.1, stable-diffusion-3

Escala
------
O protocolo integral da Secao 3.5.1 exige 180.000 imagens reais, provenientes
de COCO train2017 e LSUN. Este script opera com o subconjunto real disponivel e
registra a escala efetiva em ``corpus_info.json``, para que a reducao seja
declarada na analise dos resultados em vez de passar despercebida.

Uso::

    python scripts/organize_corpus.py --output data/corvi2024
    python scripts/organize_corpus.py --output data/corvi2024 --train-fake 5000
"""

from __future__ import annotations

import argparse
import json
import sys
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import DATA_DIR, ensure_dirs      # noqa: E402

DOWNLOAD_DIR = DATA_DIR / "downloads"

# Nome no arquivo -> identificador adotado no projeto (src/config.py)
GENERATOR_MAP = {
    # synthbuster.zip
    "glide": "glide",
    "stable-diffusion-1-3": "stable_diffusion_1_3",
    "stable-diffusion-1-4": "stable_diffusion_1_4",
    "stable-diffusion-2": "stable_diffusion_2",
    "stable-diffusion-xl": "stable_diffusion_xl",
    "dalle2": "dalle2",
    "dalle3": "dalle3",
    "firefly": "adobe_firefly",
    "midjourney-v5": "midjourney_v5",
    # extra_generators.zip
    "flux": "flux",
    "gigagan": "gigagan",
    "midjourney-v6.1": "midjourney_v6_1",
    "stable-diffusion-3": "stable_diffusion_3",
}

IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg", ".webp")


def _is_image(name: str) -> bool:
    return name.lower().endswith(IMAGE_SUFFIXES) and not name.endswith("/")


def extract_benchmark(output: Path, limit_per_generator: int | None) -> dict[str, int]:
    """Extrai os 13 geradores do benchmark para ``fake/<gerador>/``."""
    counts: dict[str, int] = {}

    for archive_name in ("synthbuster.zip", "extra_generators.zip"):
        path = DOWNLOAD_DIR / archive_name
        if not path.exists():
            print(f"[aviso] {archive_name} ausente; pulando")
            continue

        print(f"\n[{archive_name}] extraindo ...")
        with zipfile.ZipFile(path) as archive:
            # Agrupa por gerador antes de extrair, para respeitar o limite.
            by_generator: dict[str, list[str]] = {}
            for name in archive.namelist():
                if not _is_image(name):
                    continue
                parts = name.split("/")
                if len(parts) < 2:
                    continue
                # synthbuster/<gerador>/img.png  ou  <gerador>/img.png
                raw = parts[1] if parts[0] == "synthbuster" else parts[0]
                # stable-diffusion-3 vem subdividido em cfg_30/45/60
                target = GENERATOR_MAP.get(raw)
                if target is None:
                    continue
                by_generator.setdefault(target, []).append(name)

            for generator, names in sorted(by_generator.items()):
                names.sort()                      # ordem estavel entre execucoes
                selected = names if limit_per_generator is None else names[:limit_per_generator]
                destination = output / "fake" / generator
                destination.mkdir(parents=True, exist_ok=True)

                for index, name in enumerate(selected):
                    suffix = Path(name).suffix.lower()
                    target_path = destination / f"{generator}_{index:05d}{suffix}"
                    if target_path.exists():
                        continue
                    with archive.open(name) as source, target_path.open("wb") as sink:
                        sink.write(source.read())

                counts[generator] = len(selected)
                print(f"  {generator:<24} {len(selected):>6} imagens")

    return counts


def extract_training_fake(output: Path, limit: int | None) -> int:
    """Extrai as sinteticas de difusao latente usadas no treinamento."""
    path = DOWNLOAD_DIR / "latent_diffusion_trainingset.zip"
    if not path.exists():
        print("[aviso] latent_diffusion_trainingset.zip ausente; pulando")
        return 0

    print("\n[latent_diffusion_trainingset.zip] extraindo particao de treino ...")
    destination = output / "fake" / "latent_diffusion"
    destination.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(path) as archive:
        names = sorted(
            n for n in archive.namelist()
            if _is_image(n) and "/train/" in n
        )
        selected = names if limit is None else names[:limit]
        for index, name in enumerate(selected):
            target_path = destination / f"latent_diffusion_{index:06d}.png"
            if target_path.exists():
                continue
            with archive.open(name) as source, target_path.open("wb") as sink:
                sink.write(source.read())

    print(f"  latent_diffusion         {len(selected):>6} imagens")
    return len(selected)


def _official_real_list() -> list[str]:
    """Nomes das imagens reais que Corvi et al. (2024) usaram no treinamento.

    O arquivo de treino nao distribui as imagens reais, apenas a lista de nomes
    em ``real_coco.txt``. Usar essa lista -- em vez de uma amostra arbitraria do
    COCO -- garante que as imagens reais sejam exatamente as do trabalho
    original, tornando a replicacao fiel nesse aspecto.
    """
    path = DOWNLOAD_DIR / "latent_diffusion_trainingset.zip"
    if not path.exists():
        return []
    with zipfile.ZipFile(path) as archive:
        content = archive.read("latent_diffusion_trainingset/train/real_coco.txt")
    return [line.strip() for line in content.decode("utf-8").splitlines() if line.strip()]


def extract_real(output: Path, limit: int | None) -> dict[str, int]:
    """Extrai as imagens reais, preferindo COCO train2017 pela lista oficial.

    Ordem de preferencia:

    1. ``coco_train2017.zip`` filtrado pela lista oficial ``real_coco.txt`` --
       as mesmas 90.000 imagens do trabalho original.
    2. ``coco_val2017.zip`` -- alternativa de 5.000 imagens, usada enquanto o
       conjunto de treinamento nao estava disponivel.
    """
    counts: dict[str, int] = {}
    destination = output / "real" / "coco"
    destination.mkdir(parents=True, exist_ok=True)

    train_zip = DOWNLOAD_DIR / "coco_train2017.zip"
    if train_zip.exists():
        oficial = _official_real_list()
        print(f"\n[coco_train2017.zip] extraindo pela lista oficial "
              f"({len(oficial)} nomes) ...")

        with zipfile.ZipFile(train_zip) as archive:
            # Indexa por nome de arquivo para casar com a lista, que traz
            # apenas o basename.
            por_nome = {
                Path(n).name: n for n in archive.namelist() if _is_image(n)
            }
            # A ordem da lista oficial e preservada: o recorte por ``limit`` e
            # deterministico e reproduzivel entre execucoes.
            selecionados = [
                por_nome[nome] for nome in oficial if nome in por_nome
            ]
            if limit is not None:
                selecionados = selecionados[:limit]

            for index, name in enumerate(selecionados):
                target_path = destination / f"coco_{index:06d}.jpg"
                if target_path.exists():
                    continue
                with archive.open(name) as source, target_path.open("wb") as sink:
                    sink.write(source.read())

        counts["coco"] = len(selecionados)
        print(f"  coco (train2017)         {len(selecionados):>6} imagens")
        return counts

    val_zip = DOWNLOAD_DIR / "coco_val2017.zip"
    if not val_zip.exists():
        print("[aviso] nenhum arquivo do COCO encontrado; pulando")
        return counts

    print("\n[coco_val2017.zip] extraindo (alternativa reduzida) ...")
    with zipfile.ZipFile(val_zip) as archive:
        names = sorted(n for n in archive.namelist() if _is_image(n))
        selected = names if limit is None else names[:limit]
        for index, name in enumerate(selected):
            target_path = destination / f"coco_{index:06d}.jpg"
            if target_path.exists():
                continue
            with archive.open(name) as source, target_path.open("wb") as sink:
                sink.write(source.read())

    counts["coco"] = len(selected)
    print(f"  coco (val2017)           {len(selected):>6} imagens")
    return counts


def main() -> int:
    parser = argparse.ArgumentParser(description="Organiza o corpus no layout real/ e fake/")
    parser.add_argument("--output", type=Path, default=DATA_DIR / "corvi2024")
    parser.add_argument("--benchmark-per-generator", type=int, default=1000,
                        help="imagens por gerador do benchmark (padrao 1000, conforme a Secao 3.2)")
    parser.add_argument("--train-fake", type=int, default=None,
                        help="limite de sinteticas de difusao latente; padrao: todas as 180.000")
    parser.add_argument("--real", type=int, default=None,
                        help="limite de imagens reais; padrao: todas as disponiveis")
    parser.add_argument("--skip-training-set", action="store_true",
                        help="extrai apenas o benchmark de avaliacao")
    args = parser.parse_args()

    ensure_dirs()
    output = args.output
    output.mkdir(parents=True, exist_ok=True)

    fake_counts = extract_benchmark(output, args.benchmark_per_generator)
    train_fake = 0
    if not args.skip_training_set:
        train_fake = extract_training_fake(output, args.train_fake)
        if train_fake:
            fake_counts["latent_diffusion"] = train_fake
    real_counts = extract_real(output, args.real)

    total_fake = sum(fake_counts.values())
    total_real = sum(real_counts.values())

    info = {
        "output": str(output),
        "fake_por_gerador": fake_counts,
        "real_por_fonte": real_counts,
        "total_fake": total_fake,
        "total_real": total_real,
        "geradores_do_benchmark": sorted(set(GENERATOR_MAP.values())),
        "escala_reduzida": True,
        "observacao": (
            "O protocolo integral da Secao 3.5.1 preve 180.000 imagens reais "
            "(COCO train2017 + LSUN). Este corpus usa apenas as fontes reais "
            "efetivamente disponiveis; a reducao deve ser declarada na analise."
        ),
        "fontes_reais_ausentes": ["lsun", "imagenet", "raise", "fodb", "open_images"],
    }
    (output / "corpus_info.json").write_text(
        json.dumps(info, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print("\n" + "=" * 68)
    print(f"Sinteticas .......... {total_fake}")
    print(f"Reais ............... {total_real}")
    print(f"Geradores ........... {len(fake_counts)}")
    print("=" * 68)
    if total_real and total_fake / total_real > 1.5:
        print(
            f"ATENCAO: proporcao {total_fake / total_real:.1f}:1 entre sintetica e real.\n"
            "A Secao 3.5.1 preve conjunto balanceado 1:1. Use --train-fake e --real\n"
            "para equilibrar, ou aplique pesos de classe (risco R07)."
        )
    print(f"\nInformacoes gravadas em {output / 'corpus_info.json'}")
    print(f"Proximo passo:\n  python scripts/prepare_dataset.py --corpus {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
