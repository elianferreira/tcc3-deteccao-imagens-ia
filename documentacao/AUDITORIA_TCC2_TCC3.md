# Auditoria: o que o TCC 2 propôs × o que o TCC 3 entregou

Confronto etapa a etapa entre o Capítulo 3 do TCC 2 (Projeto) e o estado atual
deste repositório. Serve para que nenhuma lacuna chegue à banca sem estar
declarada.

Legenda: **✓** cumprido · **~** parcial · **✗** não feito

> **Como ler este documento.** As seções por Etapa registram a auditoria
> **original**, feita quando as lacunas foram levantadas, e são preservadas
> como histórico — várias das lacunas que elas apontam já foram fechadas desde
> então. O estado **atual** é o da tabela [Resumo das lacunas](#resumo-das-lacunas-por-prioridade),
> no fim do arquivo. Em caso de divergência, vale o resumo.

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
| T04 — Geometria projetiva | ~ três extratores funcionando; resultado negativo (AUC 0,5333) |
| T05 — Fusão | ✓ arquitetura própria |

**Divergência declarada em T03:** o TCC 2 (Seção 3.6.4) descrevia um vetor de
64 valores com uma única base e um único fator de qualidade. A implementação
final usa **540 características** (4 bases × 9 frequências × 5 fatores × 3
divergências), seguindo Bonettini et al. (2020) mais fielmente. É um avanço
sobre o previsto, já documentado no cabeçalho de `codigo/tecnicas/t03_benford/tecnica.py`.

---

## Etapa 3 — Verificação da implementação ~

Os quatro grupos de teste da Seção 3.6.2 existem:

| Grupo | Arquivo | Estado |
|---|---|---|
| Unitários | `testes/test_unit_features.py`, `test_t01_cooccurrence.py` | ✓ |
| Integração | `testes/test_integration.py` | ✓ inclui falha controlada por módulo |
| Interface | `testes/test_interface.py` | ✓ RN01, RN02, RN04, RN05, RN07 |
| Desempenho | `testes/test_performance.py` | ~ ver abaixo |

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
existe em `codigo/experimentos/runner.py:311` e a flag `--multi-seed` existe em
`automacao/run_experiments.py:285`, mas **não foram acionadas** na campanha. Sem
isso não há desvio padrão a reportar, e a estabilidade de T01 fica indemonstrada.

**Lacuna 4 — protocolo OOD do próprio TCC.** A Seção 3.6.1 define OOD como:
treinar nos **dez geradores de difusão** (Glide, SD 1.3/1.4/2/XL/3, Flux,
DALL·E 2/3, Firefly) e avaliar em **GigaGAN, Midjourney v5 e v6.1**.

O que foi executado: treinar em **um** gerador (`latent_diffusion`) e avaliar
em **doze**. É mais rigoroso, e está justificado na seção 2 de
`DECISOES_METODOLOGICAS.md` — mas **não é o protocolo do documento**. A divisão
prevista está implementada (`HELD_OUT_GENERATORS` em `codigo/configuracao.py:55`) e
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
(`automacao/testar_significancia_ood.py`) não constavam do TCC 2 e foram
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
| Pesos dos modelos | ✓ em `pesos/`, variantes preservadas |
| Documentação metodológica | ✓ acima do previsto |
| **Repositório GitHub** | **✗** o diretório **não é um repositório Git** |

**Lacuna 7 — versionamento.** A Etapa 7 prevê repositório GitHub atualizado.
Não há sequer `git init`. Todo o trabalho existe apenas localmente, sem
histórico e sem cópia remota.

---

## Resumo das lacunas, por prioridade

Situação em 30/08/2026.

| # | Lacuna | Etapa | Estado |
|---|---|---|---|
| 1 | T01 com 3 sementes (42, 123, 456) | 4 | ✓ AUC 0,9964 ± 0,0002 |
| 2 | Análise de erros | 5 | ✓ `automacao/analise_de_erros.py` |
| 3 | Grad-CAM para T01 | 5 | ✓ `automacao/gradcam_t01.py` |
| 4 | Protocolo OOD do TCC | 4 | ✓ executado; T01 0,9463 e T05 0,9261 — ver secao 3.4 de RESULTADOS.md |
| 5 | Repositório Git | 7 | ✓ github.com/elianferreira/tcc3-deteccao-imagens-ia |
| 6 | Verificação contra AUC publicada | 3 | ~ T02 conferida (+6,1 p.p. em resolução nativa); T04 contra figura, não tabela |
| 7 | RNF01 em CPU, 1.000 imagens | 3 | ✓ as quatro medidas; soma 17,871 s contra limite de 30 s |
| 8 | T04 completa | 2 | ~ os **tres** extratores rodam; resultado negativo (0,5333); **na interface desde 30/08** por servico residente no WSL2 (RNF01 20,0 s) |
| 9 | Escala integral | 1 | ~ corpus de 192.638 montado e manifestos prontos; campanha **planejada como trabalho futuro** (a decisao de 16/08 de nao executar foi reaberta em 30/08) |

### Detalhamento do que ficou parcial

**#6 — verificação contra valores publicados.**

**T04:** a conferência contra a tabela do artigo era impossível — Sarkar et al.
(2024) **não publicam tabela de AUC**. O artigo tem duas tabelas (curadoria do
corpus, e concordância entre pistas no suplementar) e reporta todo o desempenho
por curvas ROC. A leitura de figura é a única aferição disponível. Por
subconjunto (`outdoor`): duas das quatro comparações caem dentro da tolerância,
e as quatro ficam 3 a 7 p.p. **abaixo** do publicado. Detalhe na seção 4.7 de
`RESULTADOS.md`. Item encerrado por impossibilidade da fonte.

**T02 foi confrontada** com a Tabela 1 de Karageorgiou et al. (2025) —
`automacao/verificar_t02_publicado.py`, seção 4.9 de `RESULTADOS.md`. A comparação
é gerador a gerador, porque o benchmark deste trabalho usa os mesmos 13
geradores do artigo.

Resultado com duas leituras, e as duas precisam ser reportadas:

- **Sobre `glide`, o único gerador que a normalização não altera** (nativo
  256×256): AUC medida 0,9626 contra 0,9020 publicada, **+6,1 p.p.** — a
  replicação é confirmada, e supera o publicado.
- **Sobre os 12 demais:** média 0,733 contra 0,911, **−17,8 p.p.**, fora da
  tolerância. Os dois únicos dentro dela (SD 1.3 e 1.4, −3,4 e −3,3 p.p.) são os
  que menos perderam resolução.

A causa é a normalização para 256×256, adotada contra o confundidor de resolução
(lacuna resolvida em `DECISOES_METODOLOGICAS.md`, seção 4), que destrói a
evidência espectral de alta frequência de que T02 depende. As duas exigências
são incompatíveis neste corpus, e isso é achado a discutir no Capítulo 4, não
defeito a corrigir.

**#7 — RNF01. Fechada em 16/08/2026**, com as quatro técnicas medidas sobre
1.000 imagens (T02 sobre 30, por custo):

| Técnica | Dispositivo | n | Média | Desvio | Máximo | |
|---|---|---|---|---|---|---|
| T01 | CPU | 1.000 | 0,0384 s | 0,0036 s | 0,1032 s | ok |
| T03 | CPU | 1.000 | 0,0787 s | 0,0092 s | 0,1852 s | ok |
| T02 | GPU | 30 | 17,7539 s | 2,1706 s | 25,3035 s | ok |
| T05 | CPU | 1.000 | 0,0001 s | 0,0003 s | 0,0091 s | ok |
| **Soma** | | | **17,871 s** | | | **< 30 s** |

Critério **atendido**, com folga de 12 s. T02 domina o custo — é 225 vezes mais
cara que T03 —, mas mesmo assim cabe no orçamento. Resultado em
`resultados/rnf01_tempos.json`.

T04 não entra na soma: as medições acima antecedem a replicação parcial. Seu
custo é conhecido pelas extrações — 0,206 s por imagem no SSISv2 e cerca de
0,2 s no PerspectiveFields, ambos em GPU —, o que a manteria confortavelmente
dentro do limite se fosse incluída.

**Divergência interna do TCC 2 detectada.** O texto de RNF01 (Quadro 3) manda
medir com "GPU para as técnicas T02 e T04 e CPU para T01, T03 e T05". A Seção
3.6.2 manda medir **tudo** em CPU e usar GPU apenas como referência
complementar caso T02 exceda o limite. As duas especificações se contradizem.
Adotada a Seção 3.6.2, por ser a do programa de teste. **Convém uniformizar o
texto na monografia.**

**#9 — escala.** Meta do TCC: 180.000 reais + 180.000 sintéticas. Em execução:
90.000 + 90.000 + 13.000 do benchmark = 193.000. O teto de reais é 90.000
porque metade do conjunto oficial de Corvi et al. vem do LSUN, não obtido.
Proporção 1:1 mantida. Ver [`PLANO_ESCALA_INTEGRAL.md`](PLANO_ESCALA_INTEGRAL.md).

## O que foi entregue além do proposto

- Normalização do corpus após detecção de confundidor de resolução e formato
- Teste de DeLong e IC bootstrap sobre as diferenças de AUC
- Duas calibrações de T05 comparadas lado a lado
- Replicação parcial de T04 (componente objeto-sombra) sobre o corpus dos autores
- Investigação do efeito das máscaras vazias em T04 (ruído, não atalho)
- Duas variantes do protocolo OOD, que separam diversidade de treino de mudança
  de família arquitetural
- Documentação metodológica das divergências

## Erros cometidos e corrigidos durante a execução

Registrados porque cada um produziria um resultado errado na monografia se não
tivesse sido detectado.

| Erro | Consequência se não corrigido |
|---|---|
| Confundidor de resolução e formato não detectado | T03 reportada com AUC 0,985 medindo compressão, não síntese |
| Inversão medida por taxa de acerto em subconjunto de classe única | 28 inversões reportadas onde existe 1 |
| Grad-CAM alimentado com a matriz de coocorrência crua, sem o fator 1e4 | Figuras sem sentido, com 81% para toda imagem |
| Sobrescrita dos pesos do protocolo padrão pelo treino da variante OOD | Perda dos modelos que produziram os resultados documentados |
| `SyntaxError` em `run_experiments.py` não detectado antes de rodada longa | 12 h de janela perdidas |

Medida permanente adotada: `python -m compileall scripts src app tests` antes
de disparar qualquer execução longa.
