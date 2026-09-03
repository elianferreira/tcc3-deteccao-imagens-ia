"""T02 - SPAI, aprendizado espectral (Karageorgiou et al., 2025).

O que mede: assinatura espectral da imagem, por um modelo executado do
repositorio oficial em `externo/spai`.

Condicao medida: e a unica tecnica que PIORA com a normalizacao. A 256 px
ela responde 0,0% para 91,7% das reais e 80,6% das sinteticas -- nessa
resolucao o numero quase nao carrega informacao. Por isso a interface a
alimenta com resolucao nativa, com teto de 1.536 px, enquanto o escore de
256 px continua alimentando a T05 (opcao b da secao 1b).

O teto nao e cosmetico: sem ele a chamada nativa consome 5.931 MiB dos
6.144 da placa e derruba os servicos T04 do WSL2.

Resultado no protocolo padrao: AUC 0,9976, verificada contra o publicado.
Custo: subprocesso por chamada, ~20 s de arranque -- e o que viola o RNF01.

Implementacao em `tecnica.py`.
"""
from .tecnica import *          # noqa: F401,F403
