"""T04 - Geometria projetiva (Sarkar et al., 2024).

O que mede: inconsistencia geometrica da cena, por tres representacoes
independentes -- objeto-sombra (SSISv2), campos de perspectiva
(PerspectiveFields) e segmentos de reta (DeepLSD).

Os tres extratores rodam no WSL2, por servicos residentes nas portas 8404
(objeto-sombra, GPU) e 8405 (campos e retas, CPU). Um processo nao serve as
tres: o PointNet dos autores manda a identidade para CUDA sempre que a placa
esta visivel, o que obriga CUDA_VISIBLE_DEVICES="" e conflita com o
objeto-sombra. Ver armadilha 11.

Resultado: AUC 0,5333 no corpus padrao -- acaso. Mas 0,6225 sobre os
geradores modernos do benchmark, com 0,51 a 0,78 conforme o gerador. O
0,5333 mede um gerador so (latent_diffusion, 256 px). Nao e que o metodo
nao transfira; e que ele precisa de cena com geometria.

Implementacao em `tecnica.py`.
"""
from .tecnica import *          # noqa: F401,F403
