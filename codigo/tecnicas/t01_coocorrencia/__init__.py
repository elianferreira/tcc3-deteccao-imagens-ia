"""T01 - Matrizes de coocorrencia RGB + CNN (Nataraj et al., 2019).

O que mede: regularidades de textura em pares de pixels vizinhos, aprendidas
por uma CNN rasa sobre o tensor de coocorrencia dos tres canais.

Condicao medida: imagens normalizadas a 256x256. A tecnica foi TREINADA
nessa resolucao -- entregar resolucao nativa a ela inverte o resultado
(ver documentacao/ESTADO_ATUAL.md, achado 1 da interface).

Resultado no protocolo padrao: AUC 0,9962; multi-semente 0,9964 +- 0,0002.
Fragilidade conhecida: colapsa sob JPEG (0,9954 -> 0,5842 em q50).

Implementacao em `tecnica.py`.
"""
from .tecnica import *          # noqa: F401,F403
