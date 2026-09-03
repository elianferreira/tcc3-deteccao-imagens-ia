"""Campanha experimental completa sobre um corpus qualquer.

Generalizacao de ``pipeline_30k.py``, que estava preso ao corpus de 72.638
imagens. Recebe o corpus por parametro, de modo que a mesma sequencia serve
para a escala de 30k e para a de 192.638 sem duplicar codigo.

Sequencia executada
-------------------
1. Manifesto do protocolo padrao
2. Manifesto do protocolo OOD (calibracao de T05 reservada)
3. Manifesto do protocolo OOD por familias (Secao 3.6.1)
4. Treinamento e protocolo padrao
5. Multi-semente de T01, quando solicitado (Etapa 4)
6. OOD -- variante A, calibracao in-distribution
7. OOD -- variante B, calibracao held-out
8. OOD por familias
9. Protocolo de robustez

As variantes A e B do OOD sao ambas executadas de proposito: a comparacao entre
elas e resultado do trabalho, nao ajuste. Ver documentacao/DECISOES_METODOLOGICAS.md,
secao 1, e documentacao/RESULTADOS.md, secao 4.

Retomada
--------
Cada etapa e independente e os manifestos sao deterministicos, entao reexecutar
o script apos uma interrupcao refaz apenas o que faltava -- exceto o
treinamento, cuja retomada e por semente (ver ``ExperimentRunner.run_multi_seed``).

Uso::

    python automacao/pipeline_campanha.py --corpus data/corvi2024_escala --prefixo escala
    python automacao/pipeline_campanha.py --corpus data/corvi2024_escala --prefixo escala --multi-seed
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from codigo.configuracao import DATA_DIR, RESULTS_DIR      # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
PYTHON = sys.executable


def etapa(nome: str, argumentos: list[str], script: str = "run_experiments.py") -> bool:
    print("\n" + "=" * 74)
    print(f"ETAPA: {nome}")
    print("=" * 74, flush=True)

    comando = [PYTHON, "-u", str(ROOT / "automacao" / script), *argumentos]
    print(" ".join(comando) + "\n", flush=True)

    inicio = time.perf_counter()
    concluido = subprocess.run(comando, cwd=str(ROOT), check=False)
    decorrido = (time.perf_counter() - inicio) / 60

    ok = concluido.returncode == 0
    # prepare_dataset devolve 1 quando o relatorio de integridade tem ressalvas;
    # as desta campanha sao esperadas e nao impedem o prosseguimento.
    print(f"\n>>> {nome}: {'ok' if ok else f'codigo {concluido.returncode}'} "
          f"em {decorrido:.1f} min", flush=True)
    return ok


def main() -> int:
    parser = argparse.ArgumentParser(description="Campanha experimental completa")
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--prefixo", required=True,
                        help="prefixo dos manifestos, ex.: escala")
    parser.add_argument("--techniques", nargs="+", default=["T01", "T02", "T03", "T05"])
    parser.add_argument("--robustness-limit", type=int, default=1000)
    parser.add_argument("--multi-seed", action="store_true",
                        help="treina T01 com as tres sementes da Etapa 4")
    parser.add_argument("--pular-manifestos", action="store_true")
    args = parser.parse_args()

    if not args.corpus.exists():
        print(f"ERRO: corpus nao encontrado em {args.corpus}")
        return 1

    padrao = DATA_DIR / f"manifesto_{args.prefixo}_standard.csv"
    ood = DATA_DIR / f"manifesto_{args.prefixo}_ood.csv"
    familias = DATA_DIR / f"manifesto_{args.prefixo}_ood_familias.csv"
    tecnicas = list(args.techniques)
    resultados: dict[str, bool] = {}

    print("=" * 74)
    print("CAMPANHA EXPERIMENTAL")
    print("=" * 74)
    print(f"Corpus ........ {args.corpus}")
    print(f"Tecnicas ...... {tecnicas}")
    print(f"Multi-semente . {'sim' if args.multi_seed else 'nao'}")
    print(f"Resultados .... {RESULTS_DIR}", flush=True)

    # --- Manifestos -------------------------------------------------------
    if not args.pular_manifestos:
        for protocolo, destino in (("standard", padrao), ("ood", ood),
                                   ("ood_familias", familias)):
            if destino.exists():
                print(f"\n[manifesto] {destino.name} ja existe; mantido")
                continue
            etapa(f"Manifesto {protocolo}", [
                "--corpus", str(args.corpus), "--protocol", protocolo,
                "--manifest", str(destino), "--skip-duplicates",
            ], script="prepare_dataset.py")

    faltando = [p for p in (padrao, ood, familias) if not p.exists()]
    if faltando:
        print(f"ERRO: manifestos ausentes: {[p.name for p in faltando]}")
        return 1

    # --- Treinamento e protocolo padrao ----------------------------------
    argumentos = ["--protocol", "standard", "--manifest", str(padrao),
                  "--fit", "--techniques", *tecnicas]
    if args.multi_seed:
        argumentos.append("--multi-seed")
    resultados["padrao"] = etapa("Treinamento + protocolo padrao", argumentos)

    # --- OOD, duas calibracoes -------------------------------------------
    resultados["ood_calib_val"] = etapa("OOD - variante A (calibracao in-distribution)", [
        "--protocol", "ood", "--manifest", str(ood),
        "--refit-fusion", "--fusion-split", "val", "--techniques", *tecnicas,
    ])

    resultados["ood_calib_heldout"] = etapa("OOD - variante B (calibracao held-out)", [
        "--protocol", "ood", "--manifest", str(ood),
        "--refit-fusion", "--fusion-split", "fusion", "--techniques", *tecnicas,
    ])

    # --- OOD por familias, conforme a Secao 3.6.1 -------------------------
    resultados["ood_familias"] = etapa("OOD por familias (Secao 3.6.1)", [
        "--protocol", "ood", "--manifest", str(familias),
        "--refit-fusion", "--fusion-split", "val", "--techniques", *tecnicas,
    ])

    # --- Robustez ---------------------------------------------------------
    resultados["robustez"] = etapa("Protocolo de robustez", [
        "--protocol", "robustness", "--manifest", str(padrao),
        "--limit", str(args.robustness_limit), "--techniques", *tecnicas,
    ])

    print("\n" + "=" * 74)
    print("RESUMO DA CAMPANHA")
    print("=" * 74)
    for nome, ok in resultados.items():
        print(f"  {nome:<20} {'ok' if ok else 'FALHOU'}")
    print(f"\nResultados e figuras em {RESULTS_DIR}")
    return 0 if all(resultados.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
