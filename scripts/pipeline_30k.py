"""Encadeia a rodada em escala de 30k, do manifesto a campanha completa.

Aguarda a normalizacao do corpus concluir e entao executa, em sequencia:

1. Manifesto do protocolo padrao (in-distribution)
2. Manifesto do protocolo OOD, com o conjunto de calibracao de T05 reservado
3. Treinamento de T01 e T03 na nova escala
4. Protocolo padrao
5. Protocolo OOD com calibracao in-distribution   (variante A)
6. Protocolo OOD com calibracao held-out          (variante B)
7. Protocolo de robustez

As variantes A e B sao ambas executadas de proposito: a comparacao entre elas e
um resultado do trabalho, nao um detalhe de ajuste. Ver
docs/DECISOES_METODOLOGICAS.md, secao 1.

Uso::

    python scripts/pipeline_30k.py
    python scripts/pipeline_30k.py --skip-wait --robustness-limit 1000
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import DATA_DIR, RESULTS_DIR      # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
PYTHON = sys.executable
CORPUS = DATA_DIR / "corvi2024_30k_norm"
TOTAL_ESPERADO = 72_638


def aguardar_normalizacao(timeout_horas: float) -> bool:
    """Aguarda a normalizacao atingir o total esperado e estabilizar."""
    limite = time.time() + timeout_horas * 3600
    anterior = -1

    while time.time() < limite:
        if not CORPUS.exists():
            time.sleep(30)
            continue
        atual = sum(1 for _ in CORPUS.rglob("*.png"))
        if atual >= TOTAL_ESPERADO * 0.995 and atual == anterior:
            # Duas leituras iguais indicam que a gravacao terminou.
            print(f"[normalizacao] concluida: {atual} imagens")
            return True
        if atual != anterior:
            print(f"[normalizacao] {atual} de {TOTAL_ESPERADO} "
                  f"({atual / TOTAL_ESPERADO * 100:.0f}%)")
        anterior = atual
        time.sleep(60)

    print("[normalizacao] tempo esgotado")
    return False


def etapa(nome: str, argumentos: list[str], script: str = "run_experiments.py") -> bool:
    print("\n" + "=" * 74)
    print(f"ETAPA: {nome}")
    print("=" * 74, flush=True)

    comando = [PYTHON, "-u", str(ROOT / "scripts" / script), *argumentos]
    print(" ".join(comando) + "\n", flush=True)

    inicio = time.perf_counter()
    concluido = subprocess.run(comando, cwd=str(ROOT), check=False)
    decorrido = time.perf_counter() - inicio

    ok = concluido.returncode == 0
    # prepare_dataset retorna 1 quando o relatorio de integridade tem
    # ressalvas; isso nao impede o prosseguimento.
    print(f"\n>>> {nome}: {'ok' if ok else f'codigo {concluido.returncode}'} "
          f"em {decorrido / 60:.1f} min", flush=True)
    return ok


def main() -> int:
    parser = argparse.ArgumentParser(description="Rodada completa em escala de 30k")
    parser.add_argument("--techniques", nargs="+", default=["T01", "T02", "T03", "T05"])
    parser.add_argument("--robustness-limit", type=int, default=1000)
    parser.add_argument("--wait-hours", type=float, default=6.0)
    parser.add_argument("--skip-wait", action="store_true")
    args = parser.parse_args()

    print("=" * 74)
    print("RODADA EM ESCALA DE 30k - CORPUS NORMALIZADO")
    print("=" * 74)
    print(f"Corpus ........... {CORPUS}")
    print(f"Tecnicas ......... {args.techniques}")
    print(f"Resultados em .... {RESULTS_DIR}", flush=True)

    if not args.skip_wait and not aguardar_normalizacao(args.wait_hours):
        return 1

    manifesto_padrao = DATA_DIR / "manifesto30k_standard.csv"
    manifesto_ood = DATA_DIR / "manifesto30k_ood.csv"
    tecnicas = list(args.techniques)
    resultados: dict[str, bool] = {}

    # --- Manifestos -------------------------------------------------------
    etapa("Manifesto padrao", [
        "--corpus", str(CORPUS), "--protocol", "standard",
        "--manifest", str(manifesto_padrao), "--skip-duplicates",
    ], script="prepare_dataset.py")

    etapa("Manifesto OOD (com calibracao de T05 reservada)", [
        "--corpus", str(CORPUS), "--protocol", "ood",
        "--manifest", str(manifesto_ood), "--skip-duplicates",
    ], script="prepare_dataset.py")

    if not manifesto_padrao.exists() or not manifesto_ood.exists():
        print("ERRO: manifestos nao foram gerados")
        return 1

    # --- Treinamento e protocolo padrao ----------------------------------
    resultados["padrao"] = etapa("Treinamento + protocolo padrao", [
        "--protocol", "standard", "--manifest", str(manifesto_padrao),
        "--fit", "--techniques", *tecnicas,
    ])

    # --- OOD, duas calibracoes -------------------------------------------
    resultados["ood_calib_val"] = etapa("OOD - variante A (calibracao in-distribution)", [
        "--protocol", "ood", "--manifest", str(manifesto_ood),
        "--refit-fusion", "--fusion-split", "val", "--techniques", *tecnicas,
    ])

    resultados["ood_calib_heldout"] = etapa("OOD - variante B (calibracao held-out)", [
        "--protocol", "ood", "--manifest", str(manifesto_ood),
        "--refit-fusion", "--fusion-split", "fusion", "--techniques", *tecnicas,
    ])

    # --- Robustez ---------------------------------------------------------
    resultados["robustez"] = etapa("Protocolo de robustez", [
        "--protocol", "robustness", "--manifest", str(manifesto_padrao),
        "--limit", str(args.robustness_limit), "--techniques", *tecnicas,
    ])

    print("\n" + "=" * 74)
    print("RESUMO DA RODADA DE 30k")
    print("=" * 74)
    for nome, ok in resultados.items():
        print(f"  {nome:<20} {'ok' if ok else 'FALHOU'}")
    print(f"\nResultados e figuras em {RESULTS_DIR}")
    return 0 if all(resultados.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
