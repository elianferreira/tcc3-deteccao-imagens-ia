"""T05 - Arquitetura hibrida de fusao tardia (a contribuicao do trabalho).

O que faz: regressao logistica sobre o vetor de 4 escores [T01, T02, T03,
T04], com imputacao pela prevalencia quando um modulo falha (RN07/RNF04).

ATENCAO -- existem quatro modelos em `pesos/`, calibrados em regimes
diferentes, e a escolha muda o comportamento na tela:

  t05_fusion.pkl                    variante B, OOD/held-out, 3 fontes
  t05_fusion__variante_A.pkl        in-distribution, 3 fontes
  t05_fusion__quatro_fontes.pkl     padrao/val -- O QUE A INTERFACE USA
  t05_fusion__quatro_fontes_ood.pkl OOD/fusion -- recalibrado, nao usado

A interface escolhe pelo estado da T04: com os servicos no ar carrega o de
quatro fontes; sem eles, o de tres. Os dois nao foram calibrados na mesma
particao nem no mesmo protocolo -- a troca e silenciosa. Ver secao 2b.

O peso e APRENDIDO: a T02 dominar (+4,27) e artefato do regime de
calibracao, nao decisao de projeto. Recalibrado na particao fusion do OOD,
a T01 volta a ser o maior peso e a T04 troca de sinal.

Resultado no protocolo padrao: AUC 0,9996. A T04 e indiferente na fusao
(DeLong p = 0,182), o que confirma por medicao a tolerancia a fonte inutil.

Implementacao em `tecnica.py`.
"""
from .tecnica import *          # noqa: F401,F403
