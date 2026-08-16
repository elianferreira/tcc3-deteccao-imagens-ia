# Resultados experimentais — rodada 2 (corpus normalizado)

Corpus: `data/corvi2024_30k_norm` — 72.638 imagens, todas 256 × 256, PNG.
Ambiente: Python 3.11, PyTorch 2.5.1, CUDA 12.4, RTX 3060 6 GB.

Os resultados da rodada 1 (`data/corvi2024`, 22.638 imagens) estão invalidados
pelo confundidor de resolução e formato descrito na seção 4 de
[`DECISOES_METODOLOGICAS.md`](DECISOES_METODOLOGICAS.md). Eles aparecem aqui
apenas como termo de comparação.

---

## 1. Protocolo padrão (in-distribution)

Treino e teste sobre o mesmo gerador (`latent_diffusion`), conforme a Seção
3.5.1. Conjunto de teste: 9.000 imagens (4.500 reais + 4.500 sintéticas).

Arquivo: `results/resultados_standard_20260813_093714.csv`
Tempo de execução: 712,7 min (~11 h 53).

| Técnica | AUC | Acurácia | F1 | FPR | ms/imagem |
|---|---|---|---|---|---|
| T01 — Coocorrência + CNN | 0,9962 | 0,9722 | 0,9721 | 0,0233 | 5,8 |
| T02 — SPAI | 0,9976 | 0,9724 | 0,9731 | 0,0504 | 555,1 |
| T03 — Benford/DCT | 0,8064 | 0,7326 | 0,7242 | 0,2371 | 40,7 |
| **T05 — Fusão (proposta)** | **0,9996** | **0,9889** | **0,9888** | **0,0042** | — |

T05 supera a melhor técnica isolada (T02, 0,9976) e reduz a taxa de falsos
positivos em uma ordem de grandeza — de 5,04% para 0,42%. Como a interface é
declaradamente auxiliar e não substitui perícia, o custo de acusar uma imagem
real de sintética é o mais relevante na aplicação; é exatamente esse o erro que
a fusão corta.

### Observação sobre `ms_por_imagem` de T05

O valor medido para T05 é da ordem de 10⁻⁴ ms porque a fusão opera sobre
escores já calculados: mede apenas a regressão logística. **O custo real de T05
é a soma dos custos das técnicas que ela agrega** (~601 ms/imagem, dominado por
T02). Reportar o número isolado seria enganoso e ele não deve ir para a tabela
do Capítulo 4 sem essa ressalva.

---

## 2. Efeito da remoção do confundidor

Comparação direta entre as duas rodadas, mesmo protocolo padrão:

| Técnica | Rodada 1 (confundida) | Rodada 2 (normalizada) | Δ |
|---|---|---|---|
| T01 | 0,99995 | 0,99620 | −0,4 p.p. |
| T02 | 0,99934 | 0,99759 | −0,2 p.p. |
| **T03** | **0,98503** | **0,80644** | **−17,9 p.p.** |
| T05 | 0,99999 | 0,99960 | −0,04 p.p. |

### O achado

A queda concentra-se quase inteiramente em **T03**, e isso não é coincidência.

T03 aplica a Lei de Benford sobre **coeficientes DCT quantizados** — ou seja,
sua entrada é literalmente a estatística que a compressão JPEG produz. Na
rodada 1, as imagens reais vinham do COCO em JPEG e as sintéticas em PNG. T03
não estava distinguindo real de sintético: estava distinguindo **JPEG de PNG**,
com 98,5% de AUC.

Uma vez que todo o corpus passou a PNG uniforme, o atalho desapareceu e a
técnica caiu para 0,806 — valor compatível com o que a literatura reporta para
métodos baseados em Benford aplicados a imagens de difusão, e portanto
plausível como desempenho real.

T01 e T02 quase não se moveram, o que reforça a leitura: elas já operavam sobre
evidência de fato, não sobre o artefato do conjunto de dados.

### Por que isso vale para o texto

É uma demonstração empírica, dentro do próprio trabalho, do viés de conjunto de
dados discutido na introdução do TCC 2 — e com um mecanismo causal identificado,
não apenas uma suspeita. A rodada 1 não é resultado descartado: é o grupo de
controle que torna o argumento verificável.

---

## 3. Protocolo OOD

Treino em `latent_diffusion`, teste em **12 geradores nunca vistos**.
Conjunto de teste: 14.788 imagens (11.638 sintéticas + 3.150 reais).

O protocolo padrão avalia treino e teste sobre o **mesmo** gerador, de modo que
AUC próxima de 1 é esperada e **não** caracteriza capacidade de generalização.
A pergunta central do trabalho é respondida por esta seção.

### 3.1 Queda de desempenho

| Técnica | AUC padrão | AUC OOD | Queda |
|---|---|---|---|
| T01 | 0,9962 | 0,8156 | −18,1 p.p. |
| T02 | 0,9976 | 0,7365 | −26,1 p.p. |
| T03 | 0,8064 | 0,6251 | −18,1 p.p. |
| T05 | 0,9996 | 0,8228 | −17,7 p.p. |

O colapso **persiste** com 60.000 imagens de treino e sem o confundidor. A
queda de 36 a 41 p.p. medida na rodada 1 estava inflada, mas o fenômeno é real:
entre 17 e 26 p.p. Esse é o achado central do Capítulo 4.

T02 é a que mais perde, apesar de ser a mais forte in-distribution — o que
contraria a expectativa de que um modelo pré-treinado em larga escala
generalizaria melhor.

### 3.2 Significância estatística

`results/significancia_ood.json` — bootstrap estratificado (1.000 reamostragens)
e teste de DeLong para AUCs correlacionadas.

| Técnica | IC 95% da AUC |
|---|---|
| T01 | [0,8085; 0,8227] |
| T02 | [0,7277; 0,7449] |
| T03 | [0,6140; 0,6364] |
| T05 | [0,8157; 0,8298] |

| Comparação | Diferença | z | p | Conclusão |
|---|---|---|---|---|
| **T05 (A) vs T01** | +0,0113 | +2,981 | **0,0029** | **significativa** |
| T05 (A) vs T02 | +0,0903 | +20,955 | < 10⁻³⁰ | significativa |
| T05 (A) vs T03 | +0,2018 | +44,815 | < 10⁻³⁰ | significativa |
| **T05 (B) vs T01** | +0,0073 | +1,948 | **0,0514** | **não significativa** |
| T05 (B) vs T02 | +0,0863 | +19,088 | < 10⁻³⁰ | significativa |
| T05 (B) vs T03 | +0,1978 | +46,537 | < 10⁻³⁰ | significativa |

**A conclusão depende da calibração — e essa é a descoberta.**

Contra T02 e T03, a fusão vence com folga em qualquer calibração. Contra T01, a
melhor técnica isolada, o veredito se inverte conforme a variante:

- Calibrada *in-distribution* (**A**), T05 supera T01 de forma
  estatisticamente significativa (p = 0,0029) — mas opera num ponto inútil,
  deixando passar 84,5% das sintéticas (seção 4).
- Calibrada em gerador *held-out* (**B**), T05 opera num ponto utilizável, mas
  sua vantagem de ordenação sobre T01 deixa de ser demonstrável (p = 0,0514,
  com ICs sobrepostos).

Ou seja: **não existe uma configuração de T05 que seja simultaneamente
utilizável e comprovadamente superior a T01 no regime OOD.** O ganho de T05 no
protocolo padrão (seção 1) e na taxa de falsos positivos permanece válido e
significativo; é a superioridade sob mudança de distribuição que não se sustenta
sem ressalva.

Reportar apenas a variante A permitiria afirmar superioridade significativa;
reportar apenas a B levaria a descartar a contribuição. Nenhuma das duas
leituras isoladas é fiel ao que foi medido — é por isso que as duas constam.

### 3.3 Desempenho por gerador — padrão temporal

AUC no protocolo OOD, por gerador:

| Gerador | T01 | T02 | T05 |
|---|---|---|---|
| stable_diffusion_1_3 | 0,859 | 0,962 | 0,940 |
| stable_diffusion_1_4 | 0,858 | 0,963 | 0,946 |
| stable_diffusion_2 | 0,905 | 0,716 | 0,887 |
| stable_diffusion_xl | 0,936 | 0,851 | 0,913 |
| dalle2 | 0,923 | 0,695 | 0,889 |
| midjourney_v5 | 0,935 | 0,700 | 0,868 |
| gigagan | 0,873 | 0,689 | 0,832 |
| adobe_firefly | 0,844 | 0,771 | 0,788 |
| flux | 0,910 | 0,564 | 0,863 |
| midjourney_v6_1 | 0,840 | 0,626 | 0,824 |
| dalle3 | 0,563 | 0,655 | 0,620 |
| **stable_diffusion_3** | **0,350** | 0,608 | 0,552 |

Os geradores de 2022 (SD 1.3, 1.4, 2) são detectados com folga; os de 2024
(SD3, DALL·E 3) caem para perto do acaso.

**T01 no Stable Diffusion 3 marca AUC 0,350 — abaixo do acaso.** Não é
degradação, é inversão sistemática: o classificador ordena as imagens do SD3
como *mais reais* que as reais. Um detector que erra abaixo do acaso é pior que
inútil, porque suas saídas induzem a decisão errada com confiança. Merece
tratamento próprio na análise.

---

## 4. Calibração de T05: variante A × variante B

As duas variantes previstas na seção 1 de
[`DECISOES_METODOLOGICAS.md`](DECISOES_METODOLOGICAS.md), ambas sobre o mesmo
conjunto de teste OOD:

- **A** — calibração *in-distribution*, sobre a partição de validação
- **B** — calibração *held-out*, sobre `glide` + 30% das reais reservadas

| Métrica | Variante A | Variante B | Δ |
|---|---|---|---|
| AUC | 0,8269 | 0,8228 | −0,4 p.p. |
| Acurácia | 0,3343 | 0,6032 | **+26,9 p.p.** |
| F1 | 0,2684 | 0,6681 | **+40,0 p.p.** |
| Recall | 0,1552 | 0,5075 | +35,2 p.p. |
| FNR | 0,8448 | 0,4925 | −35,2 p.p. |
| FPR | 0,0038 | 0,0432 | +3,9 p.p. |

Pesos aprendidos:

| Variante | T01 | T02 | T03 |
|---|---|---|---|
| A | +3,4946 | +3,7348 | +1,0753 |
| B | +2,6707 | +2,0270 | +1,1462 |

### Leitura

A separação entre as duas métricas é o ponto. **A AUC quase não muda** (−0,4
p.p.): a calibração não altera a capacidade de ordenar imagens. O que ela muda é
o **ponto de operação**.

Calibrada in-distribution, a fusão fica excessivamente conservadora fora da
distribuição: deixa passar **84,5% das sintéticas** (FNR 0,845) para quase nunca
acusar uma real. É um detector que, na prática, quase não detecta.

Calibrada sobre gerador *held-out*, o mesmo modelo — mesma ordenação, mesma AUC
— passa a operar em um ponto utilizável: F1 sobe 40 p.p., ao custo de 3,9 p.p.
de falso positivo.

Os pesos explicam o mecanismo: em B as magnitudes caem (T02 vai de +3,73 para
+2,03), produzindo probabilidades menos extremas e um limiar de 0,5 que
corresponde a uma decisão sensata no regime OOD. O peso relativo de T03 sobe,
por ser a técnica cuja degradação in-distribution → OOD é menos abrupta.

**Conclusão metodológica:** avaliar fusão tardia com calibração
*in-distribution* subestima o método quando o alvo é generalização, e a métrica
que revela isso não é a AUC. Um trabalho que reportasse apenas AUC não veria
diferença alguma entre A e B.

---

## 4.5 Análise de erros (Etapa 5)

`scripts/analise_de_erros.py` · `results/analise_de_erros_ood.json`

### Imagens que todas as três técnicas isoladas erram

**2.993 de 14.788 (20,2%)**, com concentração muito desigual:

| Gerador | n | Todas erram | Taxa |
|---|---|---|---|
| stable_diffusion_3 | 1.000 | 660 | **66,0%** |
| dalle3 | 1.000 | 585 | **58,5%** |
| adobe_firefly | 1.000 | 334 | 33,4% |
| midjourney_v6_1 | 638 | 153 | 24,0% |
| gigagan | 1.000 | 237 | 23,7% |
| stable_diffusion_1_3 | 1.000 | 86 | 8,6% |
| **real** | 3.150 | **1** | **0,0%** |

Dois terços das imagens do Stable Diffusion 3 escapam de **todas** as
evidências simultaneamente — coocorrência de pixels, espectro e estatística
DCT. Não é uma técnica falhando: é um gerador que não deixa nenhum dos três
rastros procurados.

O contraste com a linha `real` (1 erro em 3.150) mostra que o problema é
inteiramente de falso negativo. O conjunto de técnicas quase nunca acusa uma
imagem real; ele simplesmente não enxerga as sintéticas recentes.

### Discordância — a premissa da fusão se confirma

| Par | Discordância |
|---|---|
| T01 × T02 | 38,0% |
| T01 × T03 | 38,8% |
| T02 × T03 | 43,4% |

| Técnicas que acertam | Imagens | % |
|---|---|---|
| 0 de 3 | 2.993 | 20,2% |
| 1 de 3 | 4.917 | 33,2% |
| 2 de 3 | 3.967 | 26,8% |
| 3 de 3 | 2.911 | 19,7% |

Em **60%** das imagens as técnicas divergem (1 ou 2 acertos de 3). Isso valida
empiricamente a premissa da arquitetura híbrida: as evidências são de fato
complementares, e não redundantes. Se fossem redundantes, a distribuição se
concentraria em 0 e 3.

### A fusão aproveita a complementaridade

| Entre os erros de | T05 acerta | Taxa |
|---|---|---|
| T01 (6.881 erros) | 1.351 | 19,6% |
| T02 (8.153 erros) | 3.157 | 38,7% |
| T03 (7.746 erros) | 3.647 | 47,1% |

E o custo é pequeno: em apenas **102 imagens (0,7%)** a maioria acerta e T05
erra. A fusão recupera muito mais do que estraga — este é o argumento
quantitativo mais direto a favor de T05, e independe da discussão de AUC da
seção 3.2.

### Inversão — apenas um caso, e ele é real

| Técnica | Gerador | AUC | Recall |
|---|---|---|---|
| T01 | stable_diffusion_3 | **0,3498** | 2,8% |

**Correção metodológica importante.** Uma primeira versão desta análise mediu
inversão pela taxa de acerto dentro de cada gerador e apontou 28 casos. A
medida estava errada: cada subconjunto de gerador contém **apenas imagens
sintéticas**, de modo que a taxa de acerto ali é o recall. Um limiar
conservador produz recall baixo em quase todo gerador sem que exista inversão
alguma — o classificador apenas exige mais evidência para acusar.

Inversão é outra coisa: o classificador ordenar as sintéticas de um gerador
como **mais reais que as próprias imagens reais**. Isso só aparece comparando
cada gerador contra o conjunto real, por AUC < 0,50.

Medido corretamente, há **um único caso** em todo o experimento: T01 no Stable
Diffusion 3. Os outros **29 casos** de recall abaixo de 50% são limiar
conservador, não falha de ordenação.

A distinção importa para o Capítulo 4: afirmar 28 inversões seria um erro
grosseiro de interpretação. Uma inversão isolada e severa (AUC 0,350) é um
achado bem mais específico e defensável.

---

## 4.6 Grad-CAM de T01 (Etapa 5)

`scripts/gradcam_t01.py` · figuras em `results/figuras/gradcam/`

### O que o mapa mostra — e o que não mostra

T01 **não recebe a imagem**: recebe as matrizes de coocorrência dos canais R, G
e B. Nessa representação, a posição `(i, j)` acumula quantas vezes um pixel de
intensidade `i` aparece adjacente a um pixel de intensidade `j`. **Os dois eixos
são níveis de intensidade, de 0 a 255 — não são coordenadas espaciais.**

Portanto o Grad-CAM de T01 responde *"quais transições de intensidade pesaram na
decisão"*, e não *"que região da imagem parece sintética"*. Sobrepor este mapa à
fotografia seria leitura incorreta: não existe correspondência posicional entre
os dois. Isso precisa constar da legenda da figura no Capítulo 4, porque
contraria a intuição usual de Grad-CAM sobre classificadores de pixels.

Camada alvo: `features[13]`, saída da última convolução (128 canais) após ReLU,
resolução 64 × 64 no espaço de coocorrência.

### Escores por gerador (uma imagem de cada)

| Gerador | P(sintética) |
|---|---|
| latent_diffusion (treino) | 100,0% |
| glide | 100,0% |
| dalle3 | 99,5% |
| stable_diffusion_2 | 92,7% |
| gigagan | 91,4% |
| midjourney_v5 | 72,5% |
| flux | 59,9% |
| stable_diffusion_1_3 | 20,8% |
| midjourney_v6_1 | 17,3% |
| stable_diffusion_xl | 0,3% |
| adobe_firefly, dalle2, sd_1_4, **sd_3** | ~0,0% |
| **coco (real)** | **0,0%** |

Coerente com o protocolo OOD: o gerador de treino é detectado com certeza, a
imagem real é corretamente rejeitada, e os geradores recentes passam como reais.
O Stable Diffusion 3 em 0,0% é a inversão da seção 4.5 vista em um caso
individual.

### Defeito corrigido durante a implementação

A primeira versão do script alimentava a rede com a matriz de coocorrência
**normalizada crua**, cujos valores são da ordem de 1e-5. O resultado foi
saída praticamente constante — todas as imagens, inclusive a real, recebendo
81,4% a 81,7%.

A causa: `CooccurrenceDataset` aplica um reescalonamento por `1e4` antes de
alimentar a rede, para evitar gradientes próximos de zero nas primeiras camadas.
Reproduzir o pré-processamento à mão omitiu esse fator.

A correção passou a construir a entrada pelo **próprio `CooccurrenceDataset`**,
em vez de replicar a transformação — assim o script não pode divergir do
pipeline de inferência em mudanças futuras. Registrado porque é o tipo de erro
que produziria figuras sem sentido apresentadas como resultado.

---

## 4.7 T04 — replicação parcial do classificador objeto-sombra

`scripts/replicar_t04_object_shadow.py` · corpus Kandinsky Outdoor dos autores.

**Não comparável** com T01, T02, T03 e T05: corpus diferente, não entra na
tabela principal nem na fusão. Ver seção 5 de `DECISOES_METODOLOGICAS.md`.

| Subconjunto | n | AUC | Acurácia |
|---|---|---|---|
| easy | 187.538 | 0,8293 | 0,7521 |
| unconfident | 1.410 | 0,7486 | 0,7099 |
| misclassified | 1.052 | 0,7341 | 0,6939 |
| prequalificado (união) | 2.462 | 0,7417 | 0,7031 |

### Máscaras vazias: ruído, não atalho

Durante a verificação notou-se que parte das máscaras publicadas está
**inteiramente zerada** — o SSISv2 não detectou nenhum par objeto-sombra. E a
taxa difere entre as classes:

| Classe | Máscaras vazias |
|---|---|
| Sintéticas | 34,0% |
| Reais | 44,7% |
| **Diferença** | **10,7 p.p.** |

Uma diferença sistemática entre classes levanta suspeita imediata de atalho: o
classificador poderia aprender "máscara vazia → real" e acertar sem analisar
geometria — exatamente o tipo de viés detectado na resolução e no formato do
corpus principal.

**Não é o caso.** Recalculando a AUC por grupo:

| Recorte | n | AUC |
|---|---|---|
| Todas as imagens | 2.462 | 0,7417 |
| **Apenas máscaras com conteúdo** | 1.471 | **0,8216** |
| Apenas máscaras vazias | 991 | 0,4946 |

Nas máscaras vazias a AUC é 0,4946 — o acaso, como tem de ser: a entrada é
identicamente nula e o classificador não tem o que distinguir. Elas não
carregam atalho; carregam ruído, e **rebaixam** a AUC agregada em 8 pontos.

O desempenho real do classificador sobre geometria é **0,8216**, medido nas
imagens em que o extrator efetivamente produziu representação.

### Comparação com o valor publicado (Etapa 3)

**A tabela que faltava não existe.** O artigo foi conferido inteiro: tem duas
tabelas, e nenhuma traz AUC. A Tabela 1 é a estatística de curadoria do corpus;
a Tabela 2, no material suplementar, mede concordância entre as três pistas para
SDXL. Todos os resultados quantitativos são curvas ROC (Figuras 2, 6, 10 e 11).

Isso encerra o item que constava como pendente — "conferir contra a tabela do
artigo" — por impossibilidade da fonte, não por falta de execução. A leitura de
figura é a **única** aferição disponível, e a limitação é do artigo.

Comparação por subconjunto, categoria `outdoor`, contra os valores lidos da
Figura 2:

| Subconjunto | Medido | Figura 2 | Dif. | |
|---|---|---|---|---|
| easy | 0,8293 | ~0,90 | −7,1 | fora |
| unconfident | 0,7486 | ~0,78 | −3,1 | ✓ |
| misclassified | 0,7341 | ~0,80 | −6,6 | fora |
| prequalificado (união) | 0,7417 | ~0,79 | −4,8 | ✓ |

Duas das quatro comparações ficam dentro da tolerância de 5 p.p., e **todas as
quatro ficam abaixo** do publicado — um viés sistemático de 3 a 7 p.p., não
ruído.

**Como reportar, sem exagerar para nenhum dos lados.** O critério de 5 p.p. da
Etapa 3 não pode ser aplicado com rigor aqui: os valores de referência foram
lidos de curvas, e a própria leitura carrega erro da ordem de 1 a 2 p.p. O que
se pode afirmar é que a replicação reproduz a **ordenação** e a **ordem de
grandeza** dos subconjuntos — inclusive a queda de `easy` para os difíceis, que
é o efeito central do artigo —, ficando consistentemente alguns pontos abaixo.
Afirmar "dentro da tolerância" sem essa ressalva seria escolher o recorte
favorável.

---

## 4.8 T04 sobre o corpus deste trabalho

`scripts/avaliar_t04_corpus.py` · corpus de Corvi et al. (2024), split de teste
padrão (9.000 imagens: 4.500 COCO reais, 4.500 latent diffusion).

Diferente da seção 4.7 em um ponto decisivo: ali o classificador oficial roda
sobre as máscaras que **os autores** publicaram, do corpus **deles**. Aqui as
máscaras foram extraídas neste trabalho, com o SSISv2 rodando no WSL2
(`docs/T04_AMBIENTE_WSL2.md`), sobre o corpus **deste trabalho**. É o que
permite T04 entrar na comparação.

Os pesos são os oficiais, aplicados sem reajuste, como a Etapa 2 exige. O
resultado é, portanto, de transferência direta entre domínios.

### O resultado

Split de teste, 9.000 imagens, as três variantes de pesos oficiais:

| Pesos | AUC | Acurácia | FPR |
|---|---|---|---|
| `combined` | 0,5384 | 0,4880 | 0,1627 |
| `indoor` | 0,5468 | 0,4953 | 0,1560 |
| `outdoor` | 0,4967 | 0,4637 | 0,5047 |

Praticamente o acaso nas três. Antes de aceitar o número, três verificações.

### Não é erro de mapeamento das classes

Se objeto e sombra tivessem sido trocados no agregador, a razão entre as áreas
sairia invertida:

| | Razão área objeto / área sombra |
|---|---|
| Máscaras oficiais dos autores | 5,40 |
| Extração deste trabalho | 6,72 |

Mesma direção e mesma ordem de grandeza. As classes estão corretas.

### Não é falha do extrator

A taxa de pares vazios — imagens em que o SSISv2 não detecta objeto com sombra
projetada — é o indicador mais direto de que o extrator está operando como o dos
autores:

| | Pares vazios (global) |
|---|---|
| Corpus dos autores | 36,2% |
| Extração deste trabalho | 39,1% |

Somado ao formato idêntico (binário, 256×256, massa nula em [20, 235]), o
extrator está detectando na mesma proporção.

### Não é o confundidor das máscaras vazias

A taxa de pares vazios difere entre as classes, o que levantaria a mesma
suspeita de atalho da seção 4.7 — mas com **polaridade invertida**:

| Corpus | Reais vazias | Sintéticas vazias | Direção |
|---|---|---|---|
| Kandinsky (autores) | 44,3% | 28,2% | vazio → real |
| Corvi (este trabalho) | 31,6% | 46,5% | vazio → sintética |

O atalho disponível no treino aponta para o lado oposto do que valeria aqui: um
classificador que tenha aprendido "máscara vazia → real" rotula sistematicamente
as sintéticas deste corpus como reais. Isso explica por que o desempenho fica
*em* acaso em vez de apenas cair.

O efeito do confundidor, porém, **depende da variante de pesos**, e a diferença
é grande o bastante para exigir cuidado ao citar um número. Restringindo às
5.484 imagens em que o extrator produziu representação:

| Pesos | AUC (todas) | AUC (só com conteúdo) | Variação |
|---|---|---|---|
| `combined` | 0,5384 | 0,5220 | −1,6 p.p. |
| `indoor` | 0,5468 | 0,5262 | −2,1 p.p. |
| **`outdoor`** | 0,4967 | **0,5797** | **+8,3 p.p.** |

Em `outdoor` as máscaras vazias **rebaixam** a AUC, como na seção 4.7 — e é
coerente: essa variante atribui escore alto de forma ampla (FPR 0,5047), de modo
que os pares vazios, sem informação, entram como ruído puro. Em `combined` e
`indoor` o efeito é o oposto e pequeno.

O valor estável em `outdoor` sobre pares com conteúdo se repete nos três splits:
0,5797 (teste), 0,5763 (treino), 0,5912 (validação).

### Leitura

**A comparação correta é com `outdoor`.** É a variante que a seção 4.7 usou para
confrontar o valor publicado, e o recorte comparável é o de pares com conteúdo
nos dois casos:

| | Pesos | Máscaras | Corpus | AUC (pares com conteúdo) |
|---|---|---|---|---|
| Seção 4.7 | `outdoor` | dos autores | Kandinsky | **0,8216** |
| Seção 4.8 | `outdoor` | extraídas aqui | Corvi et al. | **0,5797** |

Queda de 24 pontos. O componente objeto-sombra de T04 **não transfere** do
corpus dos autores para o de Corvi et al. (2024). O resultado em `outdoor` fica
acima do acaso — 0,58 não é 0,50 — mas muito longe do publicado, e as outras
duas variantes ficam em acaso.

É um resultado negativo, e informativo: sustenta que o desempenho geométrico
publicado depende do domínio em que foi medido — cenas Kandinsky, sintetizadas
com objetos e sombras salientes — e não se sustenta sobre um corpus de detecção
forense montado com outro critério.

**Confundidor que permanece aberto.** A comparação 0,8216 × 0,5797 muda duas
coisas ao mesmo tempo: o corpus e a origem das máscaras. Isolá-las exige rodar o
extrator deste trabalho sobre as imagens do Kandinsky e comparar com 0,8216 — o
repositório oficial distribui apenas as máscaras, não as imagens de origem, de
modo que o teste depende de baixar o dataset dos autores. Os três indícios acima
apontam para o corpus como causa, mas não substituem esse teste.

### 4.8.1 T04 como quarta fonte de T05

`scripts/pipeline_t04_fusao.py` · protocolo padrão, fusão reajustada sobre `val`.

Com os escores de T04 disponíveis, a coluna que sempre recebeu NaN em
`src/techniques/t05_fusion.py` passou a ser preenchida, e T05 foi reajustada
sobre as **quatro** fontes. Nada foi retreinado: T01 e T03 vieram dos pesos já
ajustados e os escores de T01–T03 do cache em `results/scores/`.

A pergunta não era se T04 melhora a fusão — uma fonte em acaso não tem como —,
e sim se a **degrada**. Uma entrada sem sinal pode prejudicar a decisão caso o
classificador de fusão lhe atribua peso.

| Técnica | AUC | Acurácia |
|---|---|---|
| T01 | 0,9962 | 0,9722 |
| T02 | 0,9976 | 0,9724 |
| T03 | 0,8064 | 0,7326 |
| T04 (objeto-sombra) | 0,5384 | 0,4880 |
| **T05, quatro fontes** | **0,9996** | 0,9891 |

**T05 permanece em 0,9996** — idêntica ao valor com três fontes. E o peso que a
regressão logística atribui a cada domínio explica por quê:

| Fonte | Peso |
|---|---|
| T02 (espectral) | +4,2525 |
| T01 (espacial) | +3,4061 |
| T03 (estatístico) | +0,5063 |
| **T04 (geométrico)** | **−0,0403** |

A fusão aprendeu a **ignorar** T04: peso praticamente nulo, duas ordens de
grandeza abaixo de T01 e T02. É o comportamento desejável e não estava
garantido — o classificador poderia ter se apoiado em ruído e perdido
desempenho.

Isso reforça, por evidência direta, a premissa de RN07 e RNF04: a arquitetura
tolera uma fonte inútil sem degradar a decisão. Até aqui essa tolerância havia
sido verificada apenas por desativação simulada de módulos nos testes de falha
controlada; agora foi medida com uma fonte real que de fato não carrega sinal.

O modelo de quatro fontes está em `weights/t05_fusion__quatro_fontes.pkl`.
`weights/t05_fusion.pkl` segue sendo o de três fontes, que é o que a interface
carrega.

---

## 4.9 T02 contra o valor publicado (Etapa 3)

`scripts/verificar_t02_publicado.py` · Karageorgiou et al. (2025), Tabela 1.

A comparação é direta, não aproximada: o benchmark deste trabalho usa **os
mesmos 13 geradores** da Tabela 1 do artigo. E o modelo é o oficial, sem
reajuste, treinado nas mesmas 180.000 imagens de latent diffusion de Corvi et
al. que este trabalho usa.

| Gerador | Nativo | Medido | Publicado | Dif. |
|---|---|---|---|---|
| glide | 256 px | *ver controle* | 0,902 | — |
| stable_diffusion_1_3 | 512 px | 0,962 | 0,996 | **−3,4** ✓ |
| stable_diffusion_1_4 | 512 px | 0,963 | 0,996 | **−3,3** ✓ |
| flux | 896 px | 0,564 | 0,830 | −26,6 |
| stable_diffusion_2 | 1000 px | 0,716 | 0,965 | −24,9 |
| stable_diffusion_xl | 1000 px | 0,851 | 0,974 | −12,3 |
| dalle2 | 1024 px | 0,695 | 0,911 | −21,6 |
| stable_diffusion_3 | 1024 px | 0,608 | 0,759 | −15,1 |
| gigagan | 1024 px | 0,689 | 0,854 | −16,5 |
| midjourney_v6_1 | 1040 px | 0,626 | 0,840 | −21,4 |
| dalle3 | 1080 px | 0,655 | 0,902 | −24,7 |
| midjourney_v5 | 1100 px | 0,700 | 0,945 | −24,5 |
| adobe_firefly | 2050 px | 0,771 | 0,960 | −18,9 |
| **média** | | **0,733** | **0,911** | **−17,8** |

A média fica 17,8 p.p. abaixo — muito fora da tolerância de 5 p.p. da Etapa 3.
**Não é falha de replicação**, e a própria tabela mostra por quê: os dois únicos
geradores dentro da tolerância são os dois que menos perderam resolução na
normalização do corpus.

### O controle que fecha a questão

O `glide` é nativamente 256×256, de modo que a normalização deste trabalho **não
o altera** — e é um gerador não visto no treinamento do SPAI. Ele ocupa a
partição de calibração do protocolo OOD, então seus escores já estavam
calculados.

| | Valor |
|---|---|
| n | 2.350 (1.000 glide, 1.350 reais) |
| AUC medida | **0,9626** |
| AUC publicada | 0,9020 |
| Diferença | **+6,1 p.p.** |

Em resolução nativa o valor medido **supera** o publicado. A replicação de T02 é
fiel; o desvio agregado vem da normalização.

### A tensão metodológica que isso expõe

Vale para a discussão do Capítulo 4, porque é um conflito real entre duas
exigências do próprio trabalho.

A normalização para 256×256 foi adotada na seção 4 de
`DECISOES_METODOLOGICAS.md` para remover o confundidor de resolução — sem ela, o
detector poderia separar real de sintético pelo tamanho da imagem em vez de pelo
conteúdo. Era necessária.

Mas T02 é, literalmente, *"Any-Resolution AI-Generated Image Detection by
**Spectral** Learning"*: seu sinal é a distribuição espectral, e reamostrar para
256×256 destrói justamente a evidência de alta frequência que a técnica explora.
Remover o confundidor e preservar o sinal de T02 são objetivos incompatíveis
neste corpus.

Não há dose-resposta: a correlação entre fator de redução e desvio **não** é
significativa (Spearman −0,35, p = 0,27). O efeito é de limiar — 0× fica acima
do publicado, 2× fica dentro da tolerância, e a partir de ~3,5× a degradação
satura entre −12 e −27 p.p. sem ordenação clara.

**Como reportar.** A verificação da Etapa 3 para T02 deve ser feita sobre
`glide`, o único gerador do benchmark que a normalização não afeta; nele a
replicação é confirmada. Os demais medem a técnica sob reamostragem agressiva,
que é uma condição diferente da avaliada pelos autores — e o número agregado de
0,733 deve vir sempre acompanhado dessa ressalva.

---

## 5. Protocolo de robustez

1.000 imagens in-distribution sob nove degradações.
Arquivo: `results/resultados_robustness_20260813_163751.csv` — 131,5 min.

AUC por condição:

| Condição | T01 | T02 | T03 | T05 |
|---|---|---|---|---|
| limpa | 0,9954 | 0,9982 | 0,8195 | **0,9996** |
| jpeg_q85 | 0,7530 | **0,9811** | 0,6771 | 0,9416 |
| jpeg_q70 | 0,7095 | **0,9451** | 0,6089 | 0,9032 |
| jpeg_q50 | 0,5842 | **0,8951** | 0,4415 | 0,8162 |
| ruído σ=1 | 0,9015 | **0,9982** | 0,8198 | 0,9928 |
| ruído σ=3 | 0,8123 | **0,9951** | 0,7900 | 0,9613 |
| ruído σ=5 | 0,6682 | **0,9904** | 0,7713 | 0,9175 |
| resize 85% | 0,9748 | 0,9914 | 0,8003 | **0,9949** |
| resize 70% | 0,9508 | 0,9798 | 0,7644 | **0,9856** |
| resize 50% | 0,8446 | **0,8690** | 0,6037 | 0,8149 |

### 5.1 T02 é a técnica robusta

O SPAI perde no máximo 13 p.p. (resize 50%) e mantém AUC ≥ 0,89 sob **todas**
as degradações. É o inverso do protocolo OOD, onde foi a que mais caiu. As duas
leituras juntas descrevem o método: forte contra degradação do sinal, frágil
contra mudança de gerador.

### 5.2 T01 colapsa sob compressão

De 0,9954 para 0,5842 em JPEG q50 — **41 p.p.**, praticamente o acaso. Coerente
com o mecanismo: matrizes de coocorrência medem estatística de vizinhança entre
pixels, e a quantização DCT em blocos 8 × 8 reescreve exatamente essa
vizinhança. O ruído gaussiano tem efeito semelhante, porém menor (−33 p.p. em
σ=5).

### 5.3 T03 abaixo do acaso sob JPEG

Em jpeg_q50, T03 marca **0,4415** — inversão sistemática, com FPR de 0,94.
É o mesmo fenômeno da seção 2, por outro caminho: a técnica lê coeficientes DCT
quantizados, e recomprimir em JPEG q50 reescreve esses coeficientes. Ela passa a
medir a recompressão aplicada, não a origem da imagem.

Junto com a seção 2, isso caracteriza T03 como dependente do histórico de
compressão a ponto de não ser confiável fora de condições controladas.

### 5.4 A fusão herda a fragilidade dos componentes

Sob JPEG, **T05 fica abaixo de T02 sozinha** (0,8162 contra 0,8951 em q50). Os
pesos da fusão são fixos, aprendidos em imagens limpas; quando uma degradação
destrói seletivamente dois dos três componentes, a fusão continua confiando
neles no mesmo grau.

É uma limitação estrutural da fusão tardia com pesos fixos, não um defeito de
implementação. Uma fusão sensível à condição da entrada — estimando a qualidade
antes de ponderar — é encaminhamento natural para trabalhos futuros.

---

## 6. Resumo do que a rodada estabelece

1. Sem o confundidor, T03 cai de 0,985 para 0,806: ela media compressão, não
   síntese (seção 2).
2. O colapso OOD é real, entre 17 e 26 p.p., e persiste com 60.000 imagens de
   treino (seção 3.1).
3. Detectores treinados em difusão latente de 2022 degradam para perto do acaso
   em geradores de 2024; T01 chega a **inverter** no SD3 (seção 3.3).
4. A superioridade de T05 sobre a melhor técnica isolada **não sobrevive**
   simultaneamente à exigência de calibração utilizável e ao teste estatístico
   (seção 3.2).
5. Calibração não altera AUC, altera o ponto de operação — e a diferença entre
   inútil e utilizável está aí (seção 4).
6. T02 é robusta a degradação e frágil a mudança de gerador; T01 e T03, o
   oposto sob compressão (seção 5).
7. Fusão tardia com pesos fixos herda a fragilidade dos componentes (seção 5.4).

## 7. Artefatos preservados

| Arquivo | Conteúdo |
|---|---|
| `weights/t05_fusion.pkl` | **variante B** — é a que a aplicação usa |
| `weights/t05_fusion__variante_A.pkl` | calibração in-distribution |
| `weights/t05_fusion__variante_B.pkl` | calibração held-out |
| `results/probabilidades/ood__T05__clean__variante_A.npy` | escores de A |
| `results/probabilidades/ood__T05__clean__variante_B.npy` | escores de B |
| `results/significancia_ood.json` | ICs e testes de DeLong |
| `weights/backup_22k_confundido/` | rodada 1, apenas como controle |

A aplicação usa a **variante B**. Embora ela não exiba rótulo binário (RN04), a
probabilidade mostrada só é interpretável se estiver calibrada para o regime de
uso — e o uso real é justamente sobre geradores não vistos.
