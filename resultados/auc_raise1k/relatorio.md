---
tipo: resultados
tags: [tcc3, resultados, corpus]
atualizado: 2026-09-07
---

# AUC trocando a classe real: COCO x RAISE-1k

> Gerado por `automacao/auc_raise1k_todas.py` em 07/09/2026 as 16:06. 12638 sinteticas dos 13 geradores contra 999 reais de cada fonte.
> **T04 como NaN nas duas colunas** — a T05 aqui e de tres fontes.

| tecnica | AUC com COCO | AUC com RAISE-1k | queda |
|---|---|---|---|
| **T01** | 0.8318 [0.822; 0.841] | 0.4628 [0.443; 0.484] | **-0.3690** |
| **T02** | 0.7569 [0.743; 0.770] | 0.7287 [0.715; 0.743] | **-0.0282** |
| **T03** | 0.7065 [0.688; 0.725] | 0.5618 [0.544; 0.580] | **-0.1448** |
| **T05** | 0.8597 [0.849; 0.870] | 0.5826 [0.567; 0.599] | **-0.2771** |

## Por gerador, com o RAISE como classe real

| gerador | T01 | T02 | T03 | T05 |
|---|---|---|---|---|
| adobe_firefly | 0.4414 | 0.7416 | 0.4573 | 0.5003 |
| dalle2 | 0.5796 | 0.6643 | 0.5714 | 0.6452 |
| dalle3 | 0.2507 | 0.6236 | 0.4618 | 0.2986 |
| flux | 0.5454 | 0.5320 | 0.6938 | 0.6284 |
| gigagan | 0.5049 | 0.6573 | 0.5876 | 0.5769 |
| glide | 0.6961 | 0.9450 | 0.7688 | 0.9114 |
| midjourney_v5 | 0.5531 | 0.6698 | 0.5248 | 0.6137 |
| midjourney_v6_1 | 0.4765 | 0.5956 | 0.5838 | 0.5490 |
| stable_diffusion_1_3 | 0.4255 | 0.9511 | 0.4479 | 0.6537 |
| stable_diffusion_1_4 | 0.4200 | 0.9518 | 0.4429 | 0.6517 |
| stable_diffusion_2 | 0.4496 | 0.6878 | 0.7477 | 0.6078 |
| stable_diffusion_3 | 0.1746 | 0.5760 | 0.4575 | 0.2591 |
| stable_diffusion_xl | 0.5037 | 0.8291 | 0.5656 | 0.6662 |

⚠️ **O script so mede** — a leitura de sentido entra na nota de sessao do dia.
