---
tipo: resultados
tags: [tcc3, resultados, t01, replicacao]
atualizado: 2026-09-08
---

# T01 do v1 × reimplementação independente

> Gerado em 08/09/2026. Métricas pelas mesmas funções de `codigo/metricas.py`
> que o `automacao/comparar_implementacoes_t01.py` usa.

## Por que não foi o script oficial

O `comparar_implementacoes_t01.py` **recusou as duas rotas**, e as duas recusas
estão certas:

```
--protocolo standard  ERRO: 6852 das 9000 imagens de teste nao aparecem nos
                            escores da reimplementacao.
--protocolo ood       ERRO: ood__T01__T01__clean.npy tem 7138 escores, mas a
                            particao de teste de manifesto30k_ood.csv tem
                            14788 linhas.
```

**Rota `standard`** — a réplica foi treinada no `corvi2024_escala` e o v1 no
`corvi2024_30k`. As duas partições são independentes e não aninhadas, então
grande parte do teste de um está no treino do outro. Medido nos manifestos
atuais:

| | cabeçalho do script | medido em 08/09 |
|---|---|---|
| teste 30k dentro do treino da escala | 62,7% | **76,1%** (6.852/9.000) |
| teste escala dentro do treino do 30k | 18,2% | **25,2%** (6.803/27.000) |

⚠️ **Os números do cabeçalho do script estão desatualizados** — corrigir lá.

**Rota `ood`** — o cache `ood__T01__T01__clean.npy` tem 7.138 escores e a
partição de teste de `manifesto30k_ood.csv` tem 14.788 linhas. O manifesto OOD
foi regenerado (14.788 é o n da rodada T04 de 30–31/08) e o cache da T01 é
anterior. **A comparação OOD está bloqueada até a T01 do v1 ser repontuada** —
~5,8 ms/imagem, algo como 1,5 min de GPU.

## O conjunto usado

A **interseção dos dois testes**: imagem de teste no 30k *e* de teste na escala,
logo fora do treino das duas. É a única região limpa para os dois lados.

- **2.148 imagens** — 1.487 reais (COCO) e 661 sintéticas
- Um gerador só: `latent_diffusion`. **In-distribution.**

## Resultado

| | T01 do v1 | reimplementação |
|---|---|---|
| AUC | **0,9955** | 0,9832 |
| IC 95% | [0,9933; 0,9975] | [0,9774; 0,9881] |
| Acurácia | 0,9707 | 0,9460 |
| F1 | 0,9529 | 0,9128 |
| FPR | 0,0269 | 0,0417 |
| FNR | 0,0348 | 0,0817 |

**DeLong** (AUCs correlacionadas, mesmo conjunto): diferença **0,0123**,
z = 4,2844, **p = 1,83 × 10⁻⁵**.

### Por imagem

| | |
|---|---|
| Correlação de Pearson | 0,8898 |
| Correlação de Spearman | 0,7769 |
| Concordância no limiar 0,5 | **93,81%** (133 discordâncias em 2.148) |

| Gerador | n | discordâncias | taxa | média v1 | média réplica |
|---|---|---|---|---|---|
| `latent_diffusion` | 661 | 65 | **9,83%** | 0,9612 | 0,9026 |
| `real` (COCO) | 1.487 | 68 | 4,57% | 0,0306 | 0,0488 |

## Leitura

**A técnica replica; a implementação não é intercambiável.** As duas chegam a
AUC ≥ 0,98 in-distribution, o que responde a pergunta que motivou o script: os
números da T01 no Capítulo 4 são propriedade da **técnica** de Nataraj et al.,
não de uma implementação particular.

Mas o empate não é exato. A diferença de 1,2 p.p. **é significativa**
(p = 1,8 × 10⁻⁵), e as duas discordam em 6,2% das imagens — com a discordância
**duas vezes maior nas sintéticas** (9,8%) do que nas reais (4,6%). As
distribuições também diferem: a réplica é menos extrema nas sintéticas (0,90
contra 0,96) e mais suja nas reais (0,049 contra 0,031).

### Consequência para a T05

O pickle da T05 carrega um `StandardScaler` e uma regressão logística ajustados
sobre a **distribuição de escores da T01 do v1**. Estes números dizem que
alimentar essa fusão com o escore da réplica **não se justifica**: não é o
mesmo detector, e a diferença não é de ruído.

Isso elimina o atalho — "ligar a T05 na tela v2 e pronto" —, mas não decide
entre os dois caminhos abertos:

1. trocar a linha T01 da tela v2 pelo modelo do v1, ou
2. recalibrar a T05 com os escores da réplica.

⚠️ **Esta medição é in-distribution, com um gerador só.** A pergunta da T05 é
OOD, e a comparação OOD segue bloqueada pelo cache desatualizado acima.

## Arquivos

- `escores_pareados_intersecao.csv` — os 2.148 pares, para inspeção caso a caso
