"""Etapa 1 - Preparacao do dataset.

Indexa o corpus, executa a verificacao de integridade (contagem por classe e
por gerador, confirmacao de resolucao e formato, deteccao de imagens
corrompidas e duplicatas) e grava o manifesto com o particionamento
estratificado 70/15/15.

Uso::

    python scripts/prepare_dataset.py --corpus data/corvi2024
    python scripts/prepare_dataset.py --corpus data/corvi2024 --protocol ood
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import (                                     # noqa: E402
    DATA_DIR, DIFFUSION_GENERATORS, FUSION_CALIBRATION_GENERATORS, HELD_OUT_GENERATORS,
    SEED, TRAINING_GENERATORS, ensure_dirs,
)
from src.data.manifest import (                              # noqa: E402
    assign_splits, filter_generators, format_report, index_corpus, ood_split,
    ood_split_familias, reserve_fusion_calibration, verify_integrity, write_manifest,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepara o corpus e grava o manifesto")
    parser.add_argument("--corpus", type=Path, default=DATA_DIR / "corvi2024",
                        help="raiz do corpus, com subdiretorios real/ e fake/")
    parser.add_argument("--manifest", type=Path, default=None,
                        help="caminho de saida do manifesto CSV")
    parser.add_argument("--protocol", choices=["standard", "ood", "ood_familias"],
                        default="standard",
                        help="ood: treina em um gerador e avalia nos treze restantes "
                             "(mais rigoroso). ood_familias: treina nos dez geradores de "
                             "difusao e avalia em GigaGAN e Midjourney, conforme a "
                             "Secao 3.6.1 do TCC 2")
    parser.add_argument("--no-fusion-split", action="store_true",
                        help="nao reserva o conjunto de calibracao de T05 no protocolo OOD")
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--skip-duplicates", action="store_true",
                        help="pula a deteccao de duplicatas (custosa em corpus grandes)")
    parser.add_argument("--expected-size", type=int, default=256,
                        help="resolucao esperada; use 0 para nao verificar")
    args = parser.parse_args()

    ensure_dirs()
    manifest_path = args.manifest or (DATA_DIR / f"manifesto_{args.protocol}.csv")

    print(f"Indexando corpus em {args.corpus} ...")
    entries = index_corpus(args.corpus)
    print(f"  {len(entries)} imagens encontradas\n")

    print("Verificando integridade ...")
    report = verify_integrity(
        entries,
        expected_size=args.expected_size,
        check_duplicates=not args.skip_duplicates,
    )
    print(format_report(report))

    report_path = manifest_path.with_name(f"integridade_{args.protocol}.json")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    serializable = {k: (dict(v) if hasattr(v, "most_common") else v) for k, v in report.items()}
    report_path.write_text(json.dumps(serializable, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nRelatorio gravado em {report_path}")

    if report["corrupted"]:
        print(
            f"\nATENCAO: {len(report['corrupted'])} arquivos corrompidos foram detectados "
            "e permanecem no manifesto. Remova-os do corpus antes de treinar."
        )

    print(f"\nParticionando (protocolo {args.protocol}) ...")
    if args.protocol == "standard":
        # O protocolo padrao avalia o desempenho sobre o mesmo gerador usado no
        # treinamento; os geradores do benchmark ficam reservados ao protocolo
        # OOD e nao devem contaminar esta particao.
        before = len(entries)
        entries = filter_generators(entries, TRAINING_GENERATORS)
        print(f"  restritas a {TRAINING_GENERATORS}: {before} -> {len(entries)} imagens")

    entries = assign_splits(entries, seed=args.seed)
    if args.protocol == "ood_familias":
        # Variante fiel a Secao 3.6.1: as familias GigaGAN e Midjourney ficam
        # inteiramente fora do treinamento. Nao se reserva conjunto de
        # calibracao aqui: glide, usado para isso no outro protocolo, integra o
        # treinamento nesta variante. T05 e calibrada sobre validacao.
        entries = ood_split_familias(entries)
        entries = [e for e in entries if e.split != "excluded"]
        print(f"  treino: difusao {DIFFUSION_GENERATORS[:3]}... + {TRAINING_GENERATORS}")
        print(f"  teste:  {HELD_OUT_GENERATORS}")
    elif args.protocol == "ood":
        entries = ood_split(entries)
        entries = [e for e in entries if e.split != "excluded"]
        if not args.no_fusion_split:
            # Reserva um terceiro conjunto, disjunto do treino e da avaliacao,
            # para calibrar o classificador de fusao sobre um gerador nao visto.
            entries = reserve_fusion_calibration(entries)
            print(f"  calibracao de T05 reservada: {FUSION_CALIBRATION_GENERATORS}")

    counts: dict[str, int] = {}
    for entry in entries:
        counts[entry.split] = counts.get(entry.split, 0) + 1
    for split, count in sorted(counts.items()):
        print(f"  {split:.<12} {count}")

    write_manifest(entries, manifest_path)
    print(f"\nManifesto gravado em {manifest_path}")
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
