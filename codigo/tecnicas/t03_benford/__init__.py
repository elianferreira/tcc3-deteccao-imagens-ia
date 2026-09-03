"""T03 - Lei de Benford sobre coeficientes DCT (Bonettini et al., 2020).

O que mede: divergencia entre a distribuicao do primeiro digito dos
coeficientes DCT quantizados e a lei de Benford. 540 atributos
(4 bases x 9 frequencias x 5 fatores de qualidade x 3 divergencias).

O que ela mede DE FATO: historico de compressao, nao origem da imagem.
A COCO recomprimida vai a +53,6 p.p. enquanto PNG nativo vai a -30 p.p. --
a assinatura de dupla compressao. Documentado por tres caminhos
independentes (secoes 2, 5.3 e o teste de JPEG).

Resultado no protocolo padrao: AUC 0,8064. Na interface contribui ruido
com aparencia de informacao -- acerta 0 das 5 pastas reais da varredura.

Implementacao em `tecnica.py`.
"""
from .tecnica import *          # noqa: F401,F403
