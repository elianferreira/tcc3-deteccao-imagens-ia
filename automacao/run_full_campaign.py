"""Campanha experimental completa (Etapas 4 e 5).

Executa, em sequencia, os tres protocolos da Secao 3.6.1 sobre as tecnicas
disponiveis, aguardando o treinamento de T01 concluir se ainda estiver em curso.

Ordem de execucao e motivo
--------------------------
1. Reajuste de T05 sobre os escores de validacao de TODAS as tecnicas
   disponiveis. Precisa vir primeiro: um classificador de fusao ajustado quando
   apenas uma fonte respondia aprende a decidir por ela e continua ignorando as
   demais mesmo depois que passam a responder.
2. Protocolo padrao (in-distribution).
3. Protocolo OOD -- generalizacao para os treze geradores do benchmark, nenhum
   visto no treinamento.
4. Protocolo de robustez, sobre subamostra estratificada do conjunto de teste.

Uso::

    python automacao/run_full_campaign.py
    python automacao/run_full_campaign.py --techniques T01 T02 T03 T05 --robustness-limit 1000
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from codigo.configuracao import DATA_DIR, RESULTS_DIR, WEIGHTS_DIR      # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
PYTHON = sys.executable


def wait_for_t01(timeout_hours: float) -> bool:
    """Aguarda o checkpoint de T01 aparecer, se o treinamento estiver em curso."""
    checkpoint = WEIGHTS_DIR / "t01_cooccurrence.pt"
    if checkpoint.exists():
        print(f"[T01] checkpoint presente em {checkpoint}")
        return True

    deadline = time.time() + timeout_hours * 3600
    print(f"[T01] aguardando o treinamento concluir (limite de {timeout_hours:.1f} h) ...")
    while time.time() < deadline:
        if checkpoint.exists():
            # O arquivo aparece antes de a gravacao terminar; aguarda o
            # tamanho estabilizar para nao carregar um checkpoint truncado.
            previous = -1
            while True:
                current = checkpoint.stat().st_size
                if current == previous and current > 0:
                    break
                previous = current
                time.sleep(5)
            print(f"[T01] checkpoint gravado ({checkpoint.stat().st_size / 1024**2:.1f} MB)")
            return True
        time.sleep(30)

    print(f"[T01] tempo esgotado; seguindo sem T01")
    return False


def run_step(name: str, arguments: list[str]) -> bool:
    print("\n" + "=" * 74)
    print(f"ETAPA: {name}")
    print("=" * 74)
    command = [PYTHON, str(ROOT / "automacao" / "run_experiments.py"), *arguments]
    print(" ".join(command) + "\n")

    started = time.perf_counter()
    completed = subprocess.run(command, cwd=str(ROOT), check=False)
    elapsed = time.perf_counter() - started

    status = "concluida" if completed.returncode == 0 else f"FALHOU (codigo {completed.returncode})"
    print(f"\n>>> {name}: {status} em {elapsed / 60:.1f} min")
    return completed.returncode == 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Executa a campanha experimental completa")
    parser.add_argument("--techniques", nargs="+", default=["T01", "T02", "T03", "T05"],
                        help="T04 e omitida por padrao: exige extratores geometricos "
                             "indisponiveis neste ambiente (ver externo/CONTRATO.md)")
    parser.add_argument("--robustness-limit", type=int, default=1000,
                        help="subamostra do teste no protocolo de robustez; as nove "
                             "condicoes multiplicam o custo de inferencia")
    parser.add_argument("--wait-hours", type=float, default=8.0)
    parser.add_argument("--skip-wait", action="store_true")
    args = parser.parse_args()

    print("=" * 74)
    print("CAMPANHA EXPERIMENTAL COMPLETA")
    print("=" * 74)
    print(f"Tecnicas ......... {args.techniques}")
    print(f"Resultados em .... {RESULTS_DIR}")

    if not args.skip_wait and "T01" in args.techniques:
        wait_for_t01(args.wait_hours)

    techniques = list(args.techniques)
    outcomes: dict[str, bool] = {}

    # 1 e 2. Reajuste da fusao seguido do protocolo padrao.
    outcomes["padrao"] = run_step(
        "Protocolo padrao (in-distribution) + reajuste de T05",
        ["--protocol", "standard", "--refit-fusion", "--techniques", *techniques],
    )

    # 3. Generalizacao aos treze geradores do benchmark.
    ood_manifest = DATA_DIR / "manifesto_ood.csv"
    if ood_manifest.exists():
        outcomes["ood"] = run_step(
            "Protocolo OOD (generalizacao)",
            ["--protocol", "ood", "--manifest", str(ood_manifest),
             "--techniques", *techniques],
        )
    else:
        print(f"\n[aviso] manifesto OOD ausente em {ood_manifest}; etapa pulada")
        outcomes["ood"] = False

    # 4. Robustez a degradacoes comuns.
    outcomes["robustez"] = run_step(
        "Protocolo de robustez",
        ["--protocol", "robustness", "--limit", str(args.robustness_limit),
         "--techniques", *techniques],
    )

    print("\n" + "=" * 74)
    print("RESUMO DA CAMPANHA")
    print("=" * 74)
    for name, ok in outcomes.items():
        print(f"  {name:<12} {'ok' if ok else 'FALHOU'}")
    print(f"\nResultados e figuras em {RESULTS_DIR}")
    return 0 if all(outcomes.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
