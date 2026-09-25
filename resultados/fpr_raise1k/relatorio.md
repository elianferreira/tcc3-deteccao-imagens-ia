---
tipo: resultados
tags: [tcc3, resultados, corpus]
atualizado: 2026-09-24
---

# FPR das tecnicas no RAISE-1k

> Gerado por `automacao/fpr_raise1k_todas.py` em 24/09/2026 as 01:30. Limiar 0.5.
> **A T04 entra medida nas duas amostras** — RAISE de `t04_escores_raise1k.csv` (extraido nesta rodada), COCO do cache do corpus. **Esta e a T05 de quatro fontes**, a mesma do sistema entregue, e por isso comparavel com o numero publicado.

## FPR, fonte real do treino (COCO) x fonte nova (RAISE-1k)

| tecnica | COCO (controle) | RAISE-1k | razao | publicado (padrao) |
|---|---|---|---|---|
| T01 | 0.0240 [0.016;0.035] | **0.5005** [0.470;0.531] | **20.8x** | 0.0233 |
| T02 | 0.0511 [0.039;0.067] | **0.0701** [0.056;0.088] | **1.4x** | 0.0504 |
| T03 | 0.1902 [0.167;0.216] | **0.3183** [0.290;0.348] | **1.7x** | — |
| T04 | 0.0511 [0.039;0.067] | **0.0410** [0.030;0.055] | **0.8x** | — |
| T05 | 0.0430 [0.032;0.057] | **0.4735** [0.443;0.504] | **11.0x** | 0.0042 |

## Guarda de controle

🟢 FPR da T01 no COCO: **0.0240** contra 0.0233 publicado. Dentro da ordem de grandeza — a medicao esta calibrada.

⚠️ **Leitura de sentido nao entra aqui** — o script so mede. Ver a nota de sessao do dia.
