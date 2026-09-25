"""Espera o passo 2 fechar e dispara a avaliacao do passo 3, sozinho.

Existe para que a avaliacao nao dependa da sessao continuar aberta. Roda
destacado, como o proprio passo 2.

O que ele confere antes de disparar
-----------------------------------
Terminar **nao** e o mesmo que ter terminado bem. O passo 2 pode morrer sem
traceback -- ja aconteceu neste projeto mais de uma vez --, e nesse caso o CSV
fica pela metade e a avaliacao sairia sobre um subconjunto silencioso. Entao:

1. o processo do passo 2 precisa ter saido;
2. ``t02_256_test.csv`` e ``t03_256_test.csv`` precisam cobrir **todas** as
   linhas da particao `test`, sem repetidas.

Se a cobertura estiver incompleta, ele **nao** roda a avaliacao: registra o que
falta e sai com codigo 1. Numero parcial apresentado como final e o modo de
errar que este projeto mais paga.
"""

from __future__ import annotations

import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import pandas as pd

RAIZ = Path(__file__).resolve().parents[2]
SAIDA = RAIZ / "resultados" / "t05_v2"
PYTHON = RAIZ / ".venv" / "Scripts" / "python.exe"
PID_PASSO2 = RAIZ / "logs" / "t05_v2_passo2.pid"

INTERVALO = 60          # s entre sondagens
TETO = 6 * 60 * 60      # desiste depois de 6 h


def agora() -> str:
    return datetime.now().strftime("%H:%M:%S")


def vivo(pid: int) -> bool:
    saida = subprocess.run(
        ["tasklist", "/FI", f"PID eq {pid}", "/NH"],
        capture_output=True, text=True,
    ).stdout
    return str(pid) in saida


def radical(valor: str) -> str:
    return str(valor).replace("\\", "/").split("/")[-1].rsplit(".", 1)[0].lower()


def cobertura() -> tuple[bool, str]:
    indice = pd.read_csv(SAIDA / "indice_limpeza.csv")
    esperados = {radical(p) for p in indice[indice["split"] == "test"]["path"]}
    faltas = []
    for nome in ("t02", "t03"):
        arquivo = SAIDA / f"{nome}_256_test.csv"
        if not arquivo.exists():
            faltas.append(f"{arquivo.name} ausente")
            continue
        obtidos = {radical(p) for p in pd.read_csv(arquivo)["path"]}
        faltando = esperados - obtidos
        if faltando:
            faltas.append(f"{arquivo.name}: faltam {len(faltando)} de {len(esperados)}")
    if faltas:
        return False, "; ".join(faltas)
    return True, f"{len(esperados)} imagens cobertas por T02 e T03"


def main() -> int:
    pid = int(PID_PASSO2.read_text().strip())
    print(f"[{agora()}] aguardando o passo 2 (PID {pid})", flush=True)

    limite = time.time() + TETO
    while vivo(pid):
        if time.time() > limite:
            print(f"[{agora()}] TETO DE 6 H ATINGIDO com o passo 2 ainda vivo; "
                  "nao vou avaliar", flush=True)
            return 1
        time.sleep(INTERVALO)

    print(f"[{agora()}] passo 2 encerrado; conferindo cobertura", flush=True)
    completo, detalhe = cobertura()
    print(f"[{agora()}] {detalhe}", flush=True)
    if not completo:
        print(f"[{agora()}] COBERTURA INCOMPLETA -- avaliacao NAO executada. "
              "Relance o passo 2 (ele retoma de onde parou).", flush=True)
        return 1

    print(f"[{agora()}] disparando o passo 3 (avaliacao completa)", flush=True)
    processo = subprocess.run(
        [str(PYTHON), "-u", str(RAIZ / "automacao" / "t05_v2" / "ajustar.py")],
        cwd=str(RAIZ), capture_output=True, text=True, encoding="utf-8",
        errors="replace",
    )
    print(processo.stdout, flush=True)
    if processo.stderr.strip():
        print("--- stderr ---", flush=True)
        print(processo.stderr, flush=True)

    if processo.returncode == 0:
        print(f"[{agora()}] AVALIACAO CONCLUIDA", flush=True)
    else:
        print(f"[{agora()}] AVALIACAO FALHOU (codigo {processo.returncode})",
              flush=True)
    return processo.returncode


if __name__ == "__main__":
    raise SystemExit(main())
