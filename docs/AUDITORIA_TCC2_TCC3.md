# Auditoria: o que o TCC 2 propôs × o que o TCC 3 entregou

Confronto etapa a etapa entre o Capítulo 3 do TCC 2 (Projeto) e o estado atual
deste repositório. Serve para que nenhuma lacuna chegue à banca sem estar
declarada.

Legenda: **✓** cumprido · **~** parcial · **✗** não feito

---

## Etapa 1 — Preparação do ambiente e do dataset ✓

| Item previsto | Estado |
|---|---|
| Ambiente virtual com `requirements.txt` fixado | ✓ com três pinos extras documentados |
| Clonagem dos repositórios oficiais T02 e T04 | ✓ |
| Download dos modelos pré-treinados | ✓ SPAI e geometria projetiva |
| Download e verificação de integridade do dataset | ✓ contagem por classe/gerador, resolução, corrompidos, duplicatas |
| Particionamento estratificado 70/15/15 | ✓ com alocação por maior resto |
| Sementes fixadas em 42, modo determinístico | ✓ RNF02 |

**Ressalva de escala:** o TCC 2 previa 360.000 imagens (180.000 reais +
180.000 sintéticas) de cinco bases reais (RAISE, FODB, ImageNet, COCO, Open
Images). O corpus efetivo tem **72.638 imagens** e **apenas COCO** como fonte
real. As demais exigem cadastro manual ou conta institucional. Registrado na
seção 3 de `DECISOES_METODOLOGICAS.md`.

---

## Etapa 2 — Implementação modular das técnicas ✓

Todas as técnicas expõem `fit` / `predict_proba` / `score`, com T05 dependente
das demais e isolamento de falhas (RN07).

| Técnica | Estado |
|---|---|
| T01 — Coocorrência + CNN | ✓ treinada localmente |
| T02 — SPAI | ✓ pesos oficiais, inferência por subprocesso |
| T03 — Benford/DCT + Random Forest | ✓ com busca em grade |
| T04 — Geometria projetiva | ~ módulo implementado, execução bloqueada |
| T05 — Fusão | ✓ arquitetura própria |

**Divergência declarada em T03:** o TCC 2 (Seção 3.6.4) descrevia um vetor de
64 valores com uma única base e um único fator de qualidade. A implementação
final usa **540 características** (4 bases × 9 frequências × 5 fatores × 3
divergências), seguindo Bonettini et al. (2020) mais fielmente. É um avanço
sobre o previsto, já documentado no cabeçalho de `src/techniques/t03_benford.py`.

---

## Etapa 3 — Verificação da implementação ~

Os quatro grupos de teste da Seção 3.6.2 existem:

| Grupo | Arquivo | Estado |
|---|---|---|
| Unitários | `tests/test_unit_features.py`, `test_t01_cooccurrence.py` | ✓ |
| Integração | `tests/test_integration.py` | ✓ inclui falha controlada por módulo |
| Interface | `tests/test_interface.py` | ✓ RN01, RN02, RN04, RN05, RN07 |
| Desempenho | `tests/test_performance.py` | ~ ver abaixo |

**Lacuna 1 — comparação com os valores publicados.** A Etapa 3 exige que, para
T02 e T04, a AUC obtida seja comparada à reportada nos artigos, com tolerância
de **cinco pontos percentuais**. Essa comparação **não foi executada** como
verificação formal.

**Lacuna 2 — protocolo de desempenho.** A Seção 3.6.2 especifica medição sobre
**1.000 imagens em CPU**, com média e desvio padrão, para **todas** as técnicas.
O que existe cobre T03 e T05 sobre lote reduzido. Os tempos reportados nos
resultados são de **GPU**, não de CPU.

---

## Etapa 4 — Campanha experimental ~

| Protocolo | Estado |
|---|---|
| Padrão (in-distribution) | ✓ 9.000 imagens de teste |
| OOD | ~ executado em variante **mais rigorosa** que a prevista |
| Robustez | ✓ 9 degradações (JPEG 50/70/85, ruído σ=1/3/5, resize 50/70/85%) |
| Métricas em CSV | ✓ |

**Lacuna 3 — múltiplas sementes.** A Etapa 4 determina: *"O treinamento de T01
será executado com três inicializações aleatórias distintas, utilizando as
sementes 42, 123 e 456, com resultados reportados como média e desvio padrão."*

T01 foi treinada com **uma única semente (42)**. A função `run_multi_seed`
existe em `src/experiments/runner.py:311` e a flag `--multi-seed` existe em
`scripts/run_experiments.py:285`, mas **não foram acionadas** na campanha. Sem
isso não há desvio padrão a reportar, e a estabilidade de T01 fica indemonstrada.

**Lacuna 4 — protocolo OOD do próprio TCC.** A Seção 3.6.1 define OOD como:
treinar nos **dez geradores de difusão** (Glide, SD 1.3/1.4/2/XL/3, Flux,
DALL·E 2/3, Firefly) e avaliar em **GigaGAN, Midjourney v5 e v6.1**.

O que foi executado: treinar em **um** gerador (`latent_diffusion`) e avaliar
em **doze**. É mais rigoroso, e está justificado na seção 2 de
`DECISOES_METODOLOGICAS.md` — mas **não é o protocolo do documento**. A divisão
prevista está implementada (`HELD_OUT_GENERATORS` em `src/config.py:55`) e
nunca foi rodada. Rodar as duas permitiria medir quanto do colapso decorre da
diversidade do treino.

---

## Etapa 5 — Análise comparativa ~

| Item previsto | Estado |
|---|---|
| Tabelas comparativas | ✓ |
| Curvas ROC | ✓ `roc_standard.png`, `roc_ood.png` |
| Barras de AUC por gerador | ✓ `auc_por_gerador_*.png` |
| Heatmaps de robustez | ✓ `robustez_heatmap.png` |
| AUC como métrica primária | ✓ |
| Espectros de magnitude | ✓ na interface |
| **Grad-CAM para T01** | **✗** |
| **Análise de erros** | **✗** |

**Lacuna 5 — Grad-CAM.** Previsto explicitamente: *"mapas de ativação Grad-CAM
para T01"*. Não implementado.

**Lacuna 6 — análise de erros.** Prevista: *"examinando casos em que técnicas
distintas discordam e imagens que todas as técnicas classificam incorretamente,
com agrupamento por tipo de gerador"*. Não implementada — e é justamente a
análise que daria substância à inversão de T01 no Stable Diffusion 3
(AUC 0,350) documentada em `RESULTADOS.md`, seção 3.3.

**Além do previsto:** teste de DeLong e intervalos de confiança por bootstrap
(`scripts/testar_significancia_ood.py`) não constavam do TCC 2 e foram
acrescentados — sem eles a comparação entre T05 e T01 teria sido afirmada sem
suporte estatístico.

---

## Etapa 6 — Interface ✓

| Requisito | Estado |
|---|---|
| RF01 submissão PNG/JPEG/WebP | ✓ |
| RF02 execução das técnicas | ✓ as cinco, T04 reportando indisponibilidade |
| RF03 escore percentual por técnica | ✓ |
| RF04 sumário textual | ✓ |
| RF05 espectro de magnitude | ✓ |
| RF06 validação de formato e tamanho | ✓ RN01, RN02 |
| RF07 ordem fixa das técnicas | ✓ RN05 |
| RF08 falha isolada não interrompe | ✓ RN07, verificado em teste |
| RNF03 nada persistido em disco | ✓ diretório de sessão temporário |
| RNF06 indicação de progresso | ✓ |

---

## Etapa 7 — Documentação ~

| Item | Estado |
|---|---|
| Código-fonte organizado | ✓ |
| `requirements.txt` com versões fixadas | ✓ com motivo de cada pino |
| Instruções de execução | ✓ README |
| Pesos dos modelos | ✓ em `weights/`, variantes preservadas |
| Documentação metodológica | ✓ acima do previsto |
| **Repositório GitHub** | **✗** o diretório **não é um repositório Git** |

**Lacuna 7 — versionamento.** A Etapa 7 prevê repositório GitHub atualizado.
Não há sequer `git init`. Todo o trabalho existe apenas localmente, sem
histórico e sem cópia remota.

---

## Resumo das lacunas, por prioridade

| # | Lacuna | Etapa | Esforço | Impacto |
|---|---|---|---|---|
| 1 | T01 com 3 sementes (42, 123, 456) | 4 | ~6-9 h | **Alto** — exigência textual; sem isso não há desvio padrão |
| 2 | Análise de erros | 5 | ~3 h | **Alto** — sustenta o achado da inversão no SD3 |
| 3 | Grad-CAM para T01 | 5 | ~2 h | Médio — exigência textual |
| 4 | Protocolo OOD do TCC | 4 | ~4 h | Médio — separa efeito de diversidade do treino |
| 5 | Repositório Git | 7 | ~15 min | **Alto** — risco de perda total |
| 6 | Verificação T02 × AUC publicada (5 p.p.) | 3 | ~1 h | Médio |
| 7 | RNF01 em CPU, 1.000 imagens | 3 | ~2 h | Baixo — limitação já declarável |
| 8 | T04 completa | 2 | — | Bloqueada; replicação parcial em curso |

## O que foi entregue além do proposto

- Normalização do corpus após detecção de confundidor de resolução e formato
- Teste de DeLong e IC bootstrap
- Duas calibrações de T05 comparadas lado a lado
- Replicação parcial de T04 (componente objeto-sombra) sobre o corpus dos autores
- Documentação metodológica das divergências
