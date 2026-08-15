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
