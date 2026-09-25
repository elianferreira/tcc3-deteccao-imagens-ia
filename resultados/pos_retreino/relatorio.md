---
tipo: resultados
tags: [tcc3, resultados]
atualizado: 2026-09-06
---

# Relatório pós-retreino

> Gerado automaticamente por `automacao/pos_retreino.py` em **06/09/2026 às 03:30**, sem ninguém acompanhando. As duas partes são independentes. Ver [[Plano pós-retreino]].

## Parte 1 — o retreino das 22h

`tcc3-t01-replicacao\saidas\retreino\resumo.csv` — **4 de 4 braços** concluíram.

| braco | latent_treino | semente | auc_reservados | ic_baixo | ic_alto | auc_sd3 | auc_dalle3 | auc_firefly | auc_mjv61 | minutos |
|---|---|---|---|---|---|---|---|---|---|---|
| A_balanceado | 800 | 42 | 0.8545 | 0.8472 | 0.8619 | 0.6730 | 0.9285 | 0.9232 | 0.9150 | 41.8919 |
| B_volume | 4000 | 42 | 0.8638 | 0.8561 | 0.8714 | 0.6988 | 0.9272 | 0.9211 | 0.9335 | 60.9095 |
| A_semente7 | 800 | 7 | 0.8530 | 0.8460 | 0.8597 | 0.6829 | 0.9298 | 0.9043 | 0.9190 | 44.6140 |
| A_semente123 | 800 | 123 | 0.4999 | 0.4996 | 0.5000 | 0.5000 | 0.5000 | 0.5000 | 0.4992 | 49.0240 |

### O número que decide

**`A_balanceado` nos 4 reservados: AUC 0.8545**, IC 95% [0.8472; 0.8619].
Com as 2 sementes de repetição: média **0.7358**, amplitude 0.4999–0.8545 (desvio 0.2043).

### Leitura

🟠 **Abaixo de 0.88, mas o IC inteiro está acima da T05** (0.8228). A vantagem da fusão não sobrevive como afirmação, ainda que a margem seja menor do que o limiar previa.

⚠️ **Comparação entre implementações diferentes.** A T01 da replicação não é a T01 do v1; `RESULTADO_OOD.md` registra que só o *sentido* da falha é comparável. O número acima orienta a decisão, não fecha a A3 sozinho.

- Stable Diffusion 3: **0.6730**
- DALL·E 3: **0.9285**
- Adobe Firefly: **0.9232**
- Midjourney v6.1: **0.9150**

## Parte 2 — teto da fusão condicionada à degradação

Alinhamento conferido: as 30 AUCs da Seção 5 de `RESULTADOS.md` foram reproduzidas a partir do cache antes de qualquer ajuste.

AUC por condição, avaliada nas 500 imagens que não entraram em ajuste nenhum:

| condicao | T02_sozinha | T05_so_limpa | T05_agrupada | T05_oraculo |
|---|---|---|---|---|
| clean | 0.9998 | 1.0000 | 1.0000 | 1.0000 |
| jpeg_q85 | 0.9799 | 0.9535 | 0.9525 | 0.9553 |
| jpeg_q70 | 0.9377 | 0.9095 | 0.9066 | 0.9127 |
| jpeg_q50 | 0.8891 | 0.8251 | 0.8153 | 0.8725 |
| noise_sigma1 | 0.9997 | 0.9984 | 0.9985 | 0.9989 |
| noise_sigma3 | 0.9970 | 0.9864 | 0.9901 | 0.9931 |
| noise_sigma5 | 0.9944 | 0.9681 | 0.9753 | 0.9852 |
| resize_85 | 0.9932 | 0.9971 | 0.9968 | 0.9974 |
| resize_70 | 0.9846 | 0.9885 | 0.9866 | 0.9864 |
| resize_50 | 0.8608 | 0.7882 | 0.7842 | 0.7847 |

### Porta de decisão

O oráculo supera a T02 sozinha em **0 das 3 condições de JPEG**, e ganha em média **+0.0072** de AUC sobre a prática atual (fusão ajustada só no limpo).

🔴 **Não vale construir.** Nem sabendo a condição de antemão a fusão alcança a T02 sozinha sob JPEG. A ideia morre aqui, barata, e isso é resultado negativo publicável: **quando uma fonte domina numa condição, reponderar não substitui usar só ela.**
