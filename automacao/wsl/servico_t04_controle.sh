#!/bin/bash
# Controla o servico residente de T04 dentro do WSL2.
#
# Existe porque o aninhamento de aspas entre PowerShell/Git Bash -> wsl.exe ->
# bash corrompe comandos longos. Um arquivo evita o problema por completo.
#
#   servico_t04_controle.sh iniciar [dispositivo_retas]
#   servico_t04_controle.sh saude
#   servico_t04_controle.sh log
#   servico_t04_controle.sh parar
#   servico_t04_controle.sh escore /mnt/c/caminho/da/imagem.png

set -u

RAIZ=/mnt/c/Users/ferre/projects/tcc3-deteccao-imagens-ia
PYTHON=/root/geo/bin/python
SERVICO="$RAIZ/automacao/wsl/servico_t04.py"
LOG=/tmp/servico_t04.log
PIDF=/tmp/servico_t04.pid
PORTA=8404

case "${1:-}" in
  iniciar)
    if [ -f "$PIDF" ] && kill -0 "$(cat $PIDF)" 2>/dev/null; then
      echo "ja esta rodando, pid $(cat $PIDF)"
      exit 0
    fi
    : > "$LOG"
    setsid "$PYTHON" -u "$SERVICO" \
        --porta "$PORTA" \
        --dispositivo cuda \
        --dispositivo-retas "${2:-cpu}" \
        >> "$LOG" 2>&1 < /dev/null &
    echo $! > "$PIDF"
    echo "lancado, pid $(cat $PIDF), log em $LOG"
    ;;
  saude)
    curl -s --max-time 5 "http://127.0.0.1:$PORTA/saude" || echo "sem resposta"
    echo
    ;;
  escore)
    curl -s --max-time 300 -X POST "http://127.0.0.1:$PORTA/escore" \
         -H 'Content-Type: application/json' \
         -d "{\"caminho\": \"${2:-}\"}"
    echo
    ;;
  log)
    tail -n "${2:-30}" "$LOG"
    ;;
  parar)
    if [ -f "$PIDF" ]; then
      kill "$(cat $PIDF)" 2>/dev/null && echo "encerrado pid $(cat $PIDF)"
      rm -f "$PIDF"
    else
      pkill -f servico_t04.py && echo "encerrado por nome" || echo "nao estava rodando"
    fi
    ;;
  *)
    echo "uso: $0 {iniciar|saude|escore <caminho>|log [n]|parar}"
    exit 1
    ;;
esac
