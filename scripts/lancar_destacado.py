"""Lanca um comando como processo totalmente destacado no Windows.

Motivacao
---------
As execucoes longas deste projeto -- treinamento multi-semente, montagem do
corpus em escala -- levam horas. Lancadas como filhas do shell da sessao, elas
sao encerradas junto com ele: ocorreu duas vezes, custando cerca de duas horas e
meia de treinamento de T01 na segunda.

``start /b`` do cmd tambem nao resolve: com redirecionamento de saida, o shell
permanece aguardando o filho.

Este lancador usa ``DETACHED_PROCESS`` combinado com
``CREATE_NEW_PROCESS_GROUP``, que desliga o processo do console e do grupo da
sessao. O PID e gravado para permitir acompanhamento e encerramento posteriores.

Uso::

    python scripts/lancar_destacado.py --log logs/x.log --err logs/x.err -- \\
        .venv/Scripts/python.exe -u scripts/algum_script.py --opcao valor
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

# Constantes da API do Windows; ausentes em outras plataformas.
DETACHED_PROCESS = 0x00000008
CREATE_NEW_PROCESS_GROUP = 0x00000200


def main() -> int:
    parser = argparse.ArgumentParser(description="Lanca comando destacado do shell")
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument("--err", type=Path, required=True)
    parser.add_argument("--pid", type=Path, default=None)
    parser.add_argument("--anexar", action="store_true",
                        help="acrescenta aos arquivos de saida em vez de truncar")
    parser.add_argument("comando", nargs=argparse.REMAINDER)
    args = parser.parse_args()

    comando = [a for a in args.comando if a != "--"]
    if not comando:
        print("ERRO: nenhum comando informado apos --")
        return 1

    args.log.parent.mkdir(parents=True, exist_ok=True)
    modo = "ab" if args.anexar else "wb"

    with open(args.log, modo) as saida, open(args.err, modo) as erro:
        criacao = 0
        if sys.platform == "win32":
            criacao = DETACHED_PROCESS | CREATE_NEW_PROCESS_GROUP
        processo = subprocess.Popen(
            comando,
            stdout=saida, stderr=erro, stdin=subprocess.DEVNULL,
            creationflags=criacao,
            cwd=str(Path(__file__).resolve().parent.parent),
            close_fds=True,
        )

    destino_pid = args.pid or args.log.with_suffix(".pid")
    destino_pid.write_text(str(processo.pid), encoding="ascii")

    print(f"PID {processo.pid}")
    print(f"  log: {args.log}")
    print(f"  err: {args.err}")
    print(f"  pid: {destino_pid}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
