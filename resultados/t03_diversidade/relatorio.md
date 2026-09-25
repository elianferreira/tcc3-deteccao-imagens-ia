---
tipo: resultados
tags: [tcc3, resultados, experimento]
atualizado: 2026-09-07
---

# T03 sob diversidade de geradores

> Gerado por `automacao/t03_diversidade.py` em 07/09/2026 às 04:00. Ver [[Plano pós-retreino]].

Guarda de alinhamento: as três AUCs da Seção 3.4 reproduzidas do cache. Referência da T03 a bater: **0.6956** (treinada com 55.000 imagens, 11 geradores desbalanceados).

## Passo 1 — a T03 sob diversidade

| braco | imagens | geradores | auc_test | minutos | max_depth | n_estimators | cv_auc |
|---|---|---|---|---|---|---|---|
| balanceado_11 | 18700 | 11 | 0.7281 | 10.3885 | None | 500 | 0.7844 |
| unico_ld | 18700 | 1 | 0.6368 | 10.7235 | None | 500 | 0.7960 |

**Efeito da diversidade, a volume constante: +0.0913 de AUC (+9.1 pontos).**

🟢 A alavanca que funcionou na T01 **também funciona na T03**. Vale reportar como resultado de método, não só de técnica: o defeito é do protocolo de treino com uma classe sintética, e atravessa famílias de técnica diferentes.

## Passo 2 — a fusão sobre a T03 nova

Braço adotado: **balanceado_11**.

| T05 | AUC na `test` |
|---|---|
| com a T03 atual | 0.9261 |
| com a T03 nova | 0.9243 |

Δ = **-0.0018**. ⚠️ A referência publicada da T05 é 0.9261, e a T01 sozinha faz 0.9463 nesta mesma partição — **nenhum ganho aqui reabre a manchete de AUC** enquanto a fusão não passar da T01. Ver [[T05 — Fusão tardia]].
