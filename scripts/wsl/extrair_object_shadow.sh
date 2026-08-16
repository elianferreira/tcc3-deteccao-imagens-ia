#!/usr/bin/env bash
# Extracao dos mapas objeto-sombra de T04 sobre o corpus deste trabalho.
#
# Roda dentro do WSL2, onde o detectron2 esta compilado com CUDA. Chamado pelo
# lado Windows por scripts/t04_extracao.bat, que por sua vez e disparado pelo
# Agendador de Tarefas -- ver a armadilha 2 em docs/ESTADO_ATUAL.md.
#
# A ordem das etapas e deliberada: o split de teste vem primeiro porque e o que
# basta para T04 entrar na tabela comparativa. Se a execucao for interrompida
# depois dele, o resultado principal ja esta em disco; treino e validacao, que
# so a fusao T05 exige, vem em seguida.
#
# Toda etapa e retomavel: pares ja gravados sao pulados. Os splits a processar
# podem ser passados como argumentos; sem argumentos, faz os tres.

set -u

RAIZ=/mnt/c/Users/ferre/projects/tcc3-deteccao-imagens-ia
PY=/root/geo/bin/python
EXTRATOR="$RAIZ/scripts/wsl/extrair_object_shadow_ssis.py"
SAIDA="$RAIZ/data/t04_object_shadow"

SPLITS=("$@")
if [ ${#SPLITS[@]} -eq 0 ]; then
    SPLITS=(test train val)
fi

# demo/ e o diretorio de trabalho esperado pelos caminhos relativos do SSIS.
cd /root/SSIS/demo || exit 1

for SPLIT in "${SPLITS[@]}"; do
    echo ""
    echo "######################################################################"
    echo "# split: $SPLIT"
    echo "######################################################################"
    "$PY" "$EXTRATOR" \
        --manifesto "$RAIZ/data/manifesto30k_standard.csv" \
        --split "$SPLIT" \
        --saida "$SAIDA" \
        --conjunto tcc3_30k
    echo ">>> split $SPLIT: codigo $?"
done

echo ""
echo "Extracao concluida."
