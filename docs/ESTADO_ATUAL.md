# Estado atual e como retomar

Atualizado em 02/09/2026.

> ⛔ **Leia a seção RETRATAÇÃO antes de usar qualquer número da varredura
> de pastas ou do teste de JPEG.** Quatro das cinco "fontes reais" usadas
> naquelas duas medições eram ruído procedural do corpus de teste.
>
> ✅ **A T05 foi recalibrada na tarde de 31/08** e o resultado muda a A3 —
> ver a seção **RECALIBRAÇÃO DA T05**. A interface segue com o modelo
> antigo: a troca é decisão, não conserto.
>
> 💻 **Para subir o sistema, veja `docs/COMO_RODAR.md`.** Em 02/09 o VS Code
> foi instalado e configurado; a ordem obrigatória (serviços T04 antes da
> interface) está lá em destaque.

Este arquivo existe para que o trabalho possa ser retomado em outra sessão sem
depender do histórico da conversa. Tudo que importa está no repositório.

---

## ✅ PERGUNTA RESPONDIDA — 31/08/2026

> **T04 falhava por causa do corpus inteiro, ou só por causa do
> `latent_diffusion` de 256 px?**
>
> **Resposta: era o gerador, não o método.** Sobre os geradores modernos do
> benchmark a T04 sai do acaso: **0,5333 → 0,6225**.

Rodada executada em 30–31/08/2026 sobre o split `test` de
`data/manifesto30k_ood.csv` — 14.788 imagens, 12 geradores modernos mais 3.150
reais. As três representações foram extraídas com os mesmos dispositivos da
rodada padrão (objeto-sombra em GPU, campos e retas em CPU), sob a etiqueta de
conjunto `tcc3_ood`.

### As três representações, corpus padrão × benchmark

| Representação | Corpus padrão | Benchmark | Δ |
|---|---|---|---|
| Objeto-sombra | 0,5384 | 0,5648 | +2,6 p.p. |
| Campos de perspectiva | 0,5355 | 0,5508 | +1,5 p.p. |
| Segmentos de reta | 0,5187 | **0,6234** | **+10,5 p.p.** |
| **T04 (média das três)** | **0,5333** | **0,6225** | **+8,9 p.p.** |

### AUC por gerador — é isto que responde a pergunta

Cada gerador contra as mesmas 3.150 reais. Ordenado pela média das três.

| Gerador | Objeto-sombra | Campos | Retas | Média 3 |
|---|---|---|---|---|
| dalle3 | 0,5759 | 0,6916 | 0,7814 | **0,7821** |
| flux | 0,6138 | 0,6206 | 0,7060 | 0,7164 |
| stable_diffusion_3 | 0,6155 | 0,5830 | 0,6697 | 0,6922 |
| midjourney_v6_1 | 0,6010 | 0,6192 | 0,6685 | 0,6800 |
| midjourney_v5 | 0,5528 | 0,6328 | 0,6431 | 0,6660 |
| adobe_firefly | 0,5796 | 0,5597 | 0,6358 | 0,6391 |
| stable_diffusion_xl | 0,5606 | 0,5061 | 0,6529 | 0,6159 |
| gigagan | 0,5231 | 0,5192 | 0,5821 | 0,5634 |
| stable_diffusion_1_3 | 0,5384 | 0,4912 | 0,5486 | 0,5557 |
| stable_diffusion_1_4 | 0,5357 | 0,4856 | 0,5395 | 0,5437 |
| dalle2 | 0,5326 | 0,4731 | 0,5368 | 0,5265 |
| stable_diffusion_2 | 0,5619 | 0,4527 | 0,5319 | 0,5095 |

### Como ler isto

**A ordenação é consistente nas três representações.** Não é ruído de uma
métrica: os geradores do topo estão no topo nas três, e os do fundo, no fundo.

**Resolução não explica.** DALL·E 2 e SD 2 têm ~1024 px e ficam em 0,51–0,53;
DALL·E 3, com 1080 px, chega a 0,78. Tamanho de imagem não separa os dois
grupos.

**O que separa parece ser geração e complexidade de cena.** Os geradores mais
recentes e mais fotorrealistas são os *mais* detectáveis pela geometria. É
contraintuitivo, mas coerente com o mecanismo: a T04 mede inconsistência
geométrica, e só há inconsistência onde há geometria. Modelos antigos produzem
cenas suaves e pouco estruturadas — não há o que ser inconsistente. **Isto é
hipótese consistente com o dado, não conclusão medida**; para afirmar exigiria
um controle sobre complexidade de cena.

**A estrutura disponível de fato subiu.** Mediana de 81 segmentos de reta por
imagem, contra 49 no corpus padrão (e 116 no corpus dos autores). Já o
objeto-sombra quase não mudou: 45,8% sem par, contra 46,5% antes — o que explica
por que a representação de retas ganhou 10,5 p.p. e a de objeto-sombra só 2,6.

**A média das três não supera a melhor isolada** (0,6225 contra 0,6234 das
retas). Vale registrar ao descrever a agregação.

### Efeito na redação do Capítulo 4

A afirmação *"o método geométrico não transfere"* **deixou de ser exata**. O que
o dado sustenta é mais forte e mais específico: a T04 sai do acaso em geradores
modernos, com AUC de 0,51 a 0,78 conforme o gerador, e o 0,5333 do corpus padrão
mede um gerador só. É a **A1** da pauta de orientação — que agora não pergunta
mais *"vale rodar?"*, e sim como reportar a dispersão.

### Reprodução

```powershell
scripts\t04_benchmark.bat        # 3 etapas, ~5,5 h; sobreviveu via Agendador
python scripts\avaliar_t04_corpus.py --conjunto tcc3_ood --splits test --sufixo _ood
python scripts\consolidar_escores_t04.py --conjunto tcc3_ood --sufixo _ood ^
       --saida results\t04_escores_componentes_ood.csv
python scripts\avaliar_t04_componentes.py --split test --sufixo _ood ^
       --escores results\t04_escores_componentes_ood.csv
```

Saídas em `results/t04_*_ood.*` e `data/t04_*/tcc3_ood_*`. Tempo real: 45 min de
GPU (objeto-sombra) + 203 min de CPU (campos) + 111 min de CPU (retas).

### Armadilhas de sobrescrita — agora são QUATRO

As três primeiras foram neutralizadas em 30/08; a quarta apareceu na própria
execução, em 31/08.

1. `--conjunto tcc3_ood` impede que os mapas desta rodada sobrescrevam os da
   dissertação.
2. `--sufixo _ood` em `avaliar_t04_corpus.py` protege
   `results/t04_escores_combined.csv`.
3. `--conjunto` em `consolidar_escores_t04.py` impede que ele varra o diretório
   inteiro e concatene os dois corpora em silêncio.
4. **`avaliar_t04_componentes.py` não tinha `--sufixo`** e gravava sempre em
   `results/t04_componentes_test.{json,csv}`, independente do corpus lido. Na
   primeira execução ele **sobrescreveu os números da dissertação** — recuperados
   de `git show HEAD:` e o argumento foi acrescentado ao script. Sem versionamento
   o dado teria se perdido em silêncio.

### Falhas legítimas da rodada

10 imagens de 14.788 (0,07%) sem nenhum segmento de reta detectado, listadas em
`data/t04_line_segments/tcc3_ood_ls_falhas_test.csv`. O extrator encerra com
código 1 quando há qualquer falha — **código 1 aqui não significa rodada
perdida**. Cobertura final: 99,9%.

---

## ⛔ RETRATAÇÃO — 31/08/2026, tarde

> **O "Achado 1" da varredura e a conclusão de "viés de fonte" do teste de JPEG
> estão RETIRADOS. As quatro "fontes reais" que os sustentam não são imagens
> reais — são ruído procedural.**

### Como apareceu

A sessão foi executar o próximo passo registrado no PONTO DE RETOMADA: medir a
FPR da T01 sobre as quatro fontes da amostra. O script
(`scripts/medir_fpr_t01_fontes.py`) inclui a COCO como **controle**, porque ela
deveria reproduzir a FPR de 0,0233 do Capítulo 4. O controle falhou:

| Fonte, em `data/amostra/real` | FPR medida | Mediana |
|---|---|---|
| coco | **0,6731** | 0,9057 |
| fodb | 0,7115 | 0,9792 |
| imagenet | 0,6154 | 0,8424 |
| open_images | 0,6731 | 0,8627 |
| raise | 0,6154 | 0,7761 |

A COCO da amostra é acusada tanto quanto as outras. Como a varredura media a
COCO em 0,0%, as duas COCOs não podiam ser a mesma coisa.

### A causa

**`data/amostra` é o corpus sintético de verificação funcional**, produzido por
`scripts/make_sample_corpus.py`. O `real/` dele são texturas de ruído 1/f; o
`fake/`, texturas com artefato periódico plantado. O próprio docstring do script
diz: *"metricas obtidas sobre ele NAO tem valor cientifico e nao devem ser
reportadas na monografia"*.

Quatro evidências independentes fecham:

1. **Estatística de imagem.** As cinco pastas de `real/` têm média ≈ 127, desvio
   ≈ 32 e gradiente médio ≈ 16, praticamente idênticos entre si — assinatura de
   um gerador com sementes diferentes. Fotografias variam muito: a COCO do
   corpus dá desvio 43–63 e gradiente 5–8.
2. **Contagem exata.** 52 imagens em cada uma das 5 fontes de `REAL_SOURCES` e
   20 em cada um dos 13 geradores de `ALL_GENERATORS` é precisamente o que
   `--per-class 260` produz.
3. **Falta o `latent_diffusion`.** A amostra tem 13 pastas em `fake/`, sem
   `latent_diffusion` — que não está em `ALL_GENERATORS`. O corpus real tem 14.
4. **Não existe outra origem.** `fodb`, `imagenet`, `open_images` e `raise`
   existem **só** em `data/amostra`. Os dois manifestos dão `reais por fonte:
   {'coco': 30000}`. **O corpus deste trabalho tem uma única fonte real.**

### O que exatamente cai

A varredura misturou dois corpora sem sinalizar. Das 19 pastas:

| Origem | Pastas | Situação |
|---|---|---|
| `data/corvi2024_30k` (real) | `real/coco` + os 14 geradores | **válidas** |
| `data/amostra` (sintético) | `fodb`, `imagenet`, `open_images`, `raise` | **inválidas** |

Nada avisava: os nomes das pastas são os de `REAL_SOURCES`, e o código de
varredura caiu na amostra porque é o único lugar em disco onde esses nomes
existem.

**Retirado:**

- **Achado 1 da varredura** — "a T01 acusa de sintética toda fonte real que não
  seja a COCO". As quatro linhas de 77 a 99% são ruído procedural sendo
  corretamente rejeitado por um detector. Não há viés de fonte demonstrado.
- **A conclusão "por eliminação, o viés é de fonte"** do teste de JPEG. A
  eliminação comparava a COCO real contra quatro texturas; a alternativa que
  restou nunca foi testada.
- **A coluna "Reais" do placar** das sete técnicas: 4 dos 5 itens são ruído.
  O "5/5" da T02 e o "1/5" da T01 não significam o que dizem.
- **As quatro linhas de −30 p.p.** da caracterização da T03 sob JPEG.

**Preservado, porque não depende das quatro pastas:**

- **Achado 2 — a T02 nativa é a melhor técnica.** O ganho medido está
  *inteiramente nas sintéticas* (5/14 → 11/14), e as 14 pastas de geradores vêm
  do corpus real, em resolução nativa. A opção (b) da seção 1b segue validada.
- **A sensibilidade da T01 a JPEG.** O `latent_diffusion` desabando de 100,0%
  para 0,0% em q50 é imagem real do corpus, e corrobora a seção 5.2.
- **A dupla compressão na T03.** O +53,6 p.p. da COCO é dado real e é a metade
  que importa do argumento.
- **A robustez (zero erros em 57 imagens)** e o **Achado 6** (confundidores de
  formato), que são propriedades dos arquivos.
- **Toda a rodada T04 sobre o benchmark**, que não tocou em `data/amostra`.

### O que ficou provado no lugar

A medição do controle, sobre as 30.000 COCO reais normalizadas:

| Recorte | n | FPR | IC 95% |
|---|---|---|---|
| COCO inteira | 30.000 | 0,0094 | [0,008; 0,011] |
| split `train` | 21.000 | 0,0019 | [0,001; 0,003] |
| split `val` | 4.500 | 0,0304 | [0,026; 0,036] |
| **split `test`** | **4.500** | **0,0233** | **[0,019; 0,028]** |

**O split `test` reproduz o 0,0233 do Capítulo 4 na quarta casa**, por um script
escrito do zero, lendo os arquivos e chamando `predict_proba` direto. Não é
prova de que a T01 generalize — é prova de que o número reportado é o número
medido, e de que o caminho de inferência está íntegro.

### A pergunta original continua aberta, e agora está bem posta

A FPR da T01 **é** medida contra uma única fonte real, e isso pertence às
ameaças à validade — não porque a varredura tenha mostrado viés, mas porque
**o corpus não contém outra fonte real com que comparar**. É uma limitação de
construção do corpus, verificável em uma linha (`{'coco': 30000}`), e não
depende de nenhuma medição nova para ser escrita.

Medi-la exige **baixar imagens reais de verdade** de FODB, ImageNet, Open Images
ou RAISE, normalizá-las a 256 px e rodar:

```powershell
python scripts\medir_fpr_t01_fontes.py --amostra data\<corpus real>\real
```

### As travas instaladas

Para que isto não se repita:

1. `make_sample_corpus.py` agora grava **`AVISO_CORPUS_SINTETICO.txt`** na raiz
   do corpus que produz, com os parâmetros da geração.
2. O arquivo foi criado retroativamente em `data/amostra/`.
3. `medir_fpr_t01_fontes.py` **recusa** qualquer diretório cujo caminho, ou
   algum ancestral, contenha o marcador — e recusa antes de carregar os pesos.
4. O mesmo script confere se as imagens estão em 256×256 RGB e **avisa sem
   normalizar**: normalizar em silêncio esconderia divergência do mesmo tipo.

### A lição, que vale além deste caso

**O erro não foi de execução — foi de ausência de controle.** A varredura de
19 pastas rodou sem nenhum item cuja resposta fosse conhecida de antemão, então
não tinha como acusar que quatro entradas eram de outro corpus. A medição de
hoje trazia a COCO como controle e detectou na primeira execução.

Vale como regra: **toda medição nova carrega ao menos uma entrada cujo valor já
foi publicado no capítulo.** Custa uma linha e é o que separa medir de parecer
medir.

---

## ✅ RECALIBRAÇÃO DA T05 — executada em 31/08/2026, tarde

A possibilidade registrada na seção 2b foi executada. **Funcionou, e o
resultado traz uma escolha que pertence à A3.**

### O que foi feito

| Passo | O quê | Custo real |
|---|---|---|
| 1 | Extração das três representações de T04 na partição `fusion` do OOD | 52 min |
| 2 | Classificação dos mapas objeto-sombra | 2 min |
| 3 | Consolidação e fusão dos escores num arquivo que cobre o OOD inteiro | segundos |
| 4 | Reajuste da T05 com `--fusion-split fusion` | segundos |

Scripts novos, todos versionados: `scripts/t04_glide_fusao.bat`,
`scripts/recalibrar_t05_fusion.py`, `scripts/comparar_modelos_t05.py`.

O extrator processou a partição inteira — 2.350 imagens, não só as 1.000 do
`glide` —, o que produziu um **controle de reprodução não planejado** (abaixo).

### Os pesos, lado a lado

| Modelo | T01 | T02 | T03 | T04 | Maior peso |
|---|---|---|---|---|---|
| `t05_fusion.pkl` (variante B, OOD/fusion) | +2,6707 | +2,0270 | +1,1462 | — | T01 |
| `t05_fusion__quatro_fontes.pkl` (padrão/val) | +3,4303 | **+4,2661** | +0,5205 | **−0,2512** | **T02** |
| **`t05_fusion__quatro_fontes_ood.pkl`** (novo) | **+2,6622** | +2,0252 | +1,1440 | **+0,2513** | **T01** |

**A T02 deixou de dominar.** O peso dela cai de +4,27 para +2,03 e a T01 volta a
ser a maior, exatamente como a seção 2b previa. E **a T04 troca de sinal**: de
−0,2512 para +0,2513 — calibrada num cenário onde as fontes erram, ela deixa de
ser penalizada.

### As duas bancadas

`scripts/comparar_modelos_t05.py` alimenta os três modelos com **a mesma matriz
de escores**. Nada é reajustado; a única variável é o modelo. É o cuidado que a
armadilha nº 6 exige.

**Bancada 1 — split `test` do protocolo padrão, n = 9.000, in-distribution**

| Modelo | AUC | Acurácia | FPR | FNR |
|---|---|---|---|---|
| variante B | 0,9992 | 0,9787 | 0,0400 | 0,0027 |
| quatro fontes (padrão/val) | **0,9996** | **0,9889** | **0,0044** | 0,0178 |
| **quatro fontes (OOD/fusion)** | 0,9992 | 0,9787 | 0,0400 | **0,0027** |

**Bancada 2 — a varredura, só as 15 pastas válidas**

As quatro pastas de ruído da RETRATAÇÃO ficam de fora. Restam a COCO e os 14
geradores.

| Modelo | Pastas certas | Reais | Sintéticas |
|---|---|---|---|
| variante B | **7/15** | 1/1 | **6/14** |
| quatro fontes (padrão/val) — **o de produção** | 3/15 | 1/1 | 2/14 |
| **quatro fontes (OOD/fusion)** | **7/15** | 1/1 | **6/14** |

Mediana por pasta, em % de síntese. `*` marca erro.

| Pasta | variante B | produção | **recalibrado** |
|---|---|---|---|
| real/coco | 2,6 | 0,0 | 2,3 |
| fake/gigagan | 67,2 | 33,0* | **74,8** |
| fake/midjourney_v5 | 65,5 | 4,1* | **72,8** |
| fake/stable_diffusion_2 | 72,2 | 8,1* | **64,4** |
| fake/stable_diffusion_1_3 | 66,3 | 35,4* | **64,7** |
| fake/glide | 99,8 | 99,5 | 99,8 |
| fake/latent_diffusion | 100,0 | 99,9 | 100,0 |
| fake/flux | 35,6* | 0,5* | 36,2* |
| fake/stable_diffusion_xl | 28,8* | 0,1* | 28,1* |
| fake/stable_diffusion_1_4 | 17,8* | 16,3* | 18,5* |
| fake/midjourney_v6_1 | 10,1* | 0,0* | 11,2* |
| fake/dalle3 | 3,6* | 0,0* | 3,0* |
| fake/stable_diffusion_3 | 1,7* | 0,0* | 2,4* |
| fake/dalle2 | 0,8* | 0,0* | 1,1* |
| fake/adobe_firefly | 0,9* | 0,0* | 0,7* |

### Achado 1 — o recalibrado é a variante B com um termo a mais

Não é interpretação, é medição sobre as 9.000 da bancada 1:

| | |
|---|---|
| Correlação com a variante B | **0,99967** |
| Diferença mediana | 6,8e-04 |
| Decisões divergentes | **22 de 9.000** |

Os pesos de T01, T02 e T03 batem na terceira casa com os da variante B. **O que
determinou o comportamento foi a partição de calibração, não a quarta fonte.**
A T04 entra com +0,2513 e move quase nada — coerente com o **p = 0,182** do
teste de DeLong, e agora medido por um segundo caminho.

Isso **fortalece** o registro da seção 4.11: a arquitetura tolera fonte inútil,
e a T04 continua sendo fonte inútil neste corpus.

### Achado 2 — o conserto tem um preço, e ele cai em cima da A3

O recalibrado acerta na tela e **piora o número de maior destaque do capítulo**:

| | Produção (padrão/val) | Recalibrado (OOD/fusion) |
|---|---|---|
| Falsos positivos, split `test` | **20 de 4.500** | **180 de 4.500** |
| FPR | **0,0044** | **0,0400** |

A pauta de orientação lista, na **A3**, como candidata a contribuição da T05:
*"queda de falsos positivos 5,04% → 0,42%"*. Os 5,04% são a FPR da T02, a melhor
fonte isolada; os 0,42% são a FPR da T05 **de produção**.

Com o modelo recalibrado essa linha vira **5,04% → 4,00%**, e a contribuição
praticamente desaparece.

**Ou seja: os dois candidatos da A3 são mutuamente exclusivos.**

- Quem quiser a **queda de falsos positivos** como contribuição fica com o
  modelo de produção — e aceita que a tela erre em gerador moderno.
- Quem quiser a **fusão que funciona fora da distribuição** fica com o
  recalibrado — e troca a manchete da A3 por outra.

Isto não é conserto de interface. **É a A3, agora com número dos dois lados.**

### Achado 3 — a cadeia de extração de T04 reproduz

O extrator processou as 1.350 reais da partição `fusion` junto com as 1.000 do
`glide`. Como essas 1.350 já tinham escore desde a dissertação, a reextração
virou controle:

| | |
|---|---|
| Imagens em comum | 1.350 |
| Idênticas bit a bit | **1.309 (97,0%)** |
| Diferença máxima | **9,35e-05** |
| Diferença mediana | 0,0 |

Dentro do resíduo de cuDNN de 2,3e-04 já documentado para lote 1 × lote 128. **A
cadeia de extração de T04 reproduz** — é a primeira verificação disso desde que
os extratores foram postos de pé.

### O que foi preservado, e por quê

**Nenhum modelo reportado foi tocado.** O reajuste grava em
`weights_t05_ood_fusion/` e o resultado vai para `weights/` sob nome próprio,
`t05_fusion__quatro_fontes_ood.pkl`. A interface **continua carregando o de
produção** — trocar isso é decisão, não conserto.

Três armadilhas apareceram no caminho e valem registro:

1. **A nº 4 quase mordeu.** O comando registrado na seção 2b —
   `run_experiments.py --protocol ood --refit-fusion --fusion-split fusion` —
   grava em `WEIGHTS_DIR/t05_fusion.pkl`, que é **a variante B que a interface
   usa**. Rodá-lo sem `TCC3_WEIGHTS_DIR` destruiria um modelo reportado. O
   `recalibrar_t05_fusion.py` isola.
2. **A nº 7 mordeu de verdade.** Os caches `ood__*__clean` (7.138) e
   `ood__*__val` (10.500) são do `manifesto30k_ood_familias.csv`, não do
   `manifesto30k_ood.csv` (14.788 e 9.000). A guarda de forma os rejeitaria e a
   T02 seria recomputada em 14.788 imagens — **mais de duas horas**. Por isso o
   script para depois de gravar o modelo, e a avaliação completa do protocolo
   OOD **continua em aberto**, com custo declarado.
3. **A nº 5 não custou nada aqui:** só a origem dos escores de T04 mudou, e não
   havia nenhum `ood__T04__*` em cache. Os de T01/T02/T03 foram preservados de
   propósito — apagá-los é que teria custado as duas horas.

### Uma falha pré-existente, para não assustar quem reproduzir

`coco_022232` não tem escore de T04 em rodada nenhuma: é falha do extrator
objeto-sombra desde a dissertação (`KeyError: np.int64(12)`, registrada em
`data/t04_object_shadow/tcc3_30k_falhas_train_*.csv`). Fica no split `train`, que
o reajuste não consome. O `recalibrar_t05_fusion.py` exige 100% na partição
`fusion` e apenas **avisa** nas demais, para que uma falha antiga e conhecida não
bloqueie uma operação que não depende dela.

Na extração nova houve **uma** falha legítima: `coco_029494`, sem retas
detectadas. Cobertura 2.349/2.350.

### O que fica em aberto

- **A escolha da A3**, acima — é a decisão que importa e é do orientador.
- **A avaliação do recalibrado no split `test` do OOD** (14.788 imagens), que
  daria a AUC honesta do modelo novo. Custa as ~2 h de recomputação de T02
  descritas na armadilha nº 7.
- **Trocar o modelo que a interface carrega**, em `DetectionService._load_all()`.
  Só faz sentido depois da A3.

---

## Como retomar em uma sessão nova

Cole isto numa sessão nova:

> Estou continuando meu TCC 3 em `C:\Users\ferre\projects\tcc3-deteccao-imagens-ia`.
> Leia `docs/ESTADO_ATUAL.md` inteiro antes de responder qualquer coisa, e
> confira o que ainda está vivo antes de supor progresso.
>
> As sessões de 31/08 e 02/09 fizeram isto, tudo já registrado — **não refaça**:
>
> - normalizaram o envio da interface (item 1) e puseram a T02 em resolução
>   nativa com teto de 1.536 px (item 1b, opção b);
> - rodaram a varredura das 19 pastas e o teste de JPEG, e depois **retrataram
>   parte dos dois** — leia a seção **RETRATAÇÃO** antes de qualquer outra
>   coisa. Quatro das cinco "fontes reais" daquelas medições eram ruído
>   procedural de `data/amostra`;
> - **recalibraram a T05** na partição `fusion` do OOD. Funciona (3/15 → 7/15
>   pastas) mas custa a FPR de 0,0044 — os dois candidatos a contribuição da A3
>   se anulam. Ver **RECALIBRAÇÃO DA T05**;
> - instalaram e configuraram o **VS Code**. Ver `docs/COMO_RODAR.md`.
>
> A investigação da T01 está **encerrada por falta de dado**: o corpus tem uma
> única fonte real, a COCO. O que sobrou dela é uma frase para as ameaças à
> validade, que não precisa de medição.
>
> **Nada foi commitado desde `42aae24` — são 41 arquivos em três sessões.**
> Confira `git status` antes de mexer, e considere commitar antes de qualquer
> outra coisa.

**Em 02/09/2026 não há nada em execução além dos serviços T04**, nas portas
8404 e 8405. A **interface está parada de propósito** — foi derrubada para ser
subida pelo VS Code. Como os serviços T04 já estão no ar, a ordem obrigatória
está satisfeita e a interface pode subir direto (`docs/COMO_RODAR.md`).

> ⚠️ **Ordem de subida: serviços T04 primeiro, interface depois.** A
> `is_available()` da T04 roda na carga do Gradio; subir a interface antes dos
> serviços marca a T04 como indisponível pela sessão inteira, e a tela troca de
> modelo de fusão por causa disso (ver seção 2b).

Se uma sessão futura deixar algo rodando, confirme antes de supor
progresso — o log de uma execução interrompida para no meio sem marca de erro:

```powershell
Get-Process python,pythonw | Select-Object Id,StartTime
schtasks /query /fo list | Select-String "tcc3_"
Get-NetTCPConnection -State Listen | Where-Object LocalPort -in 7860,8404,8405
wsl -d Ubuntu-24.04 -u root -- bash -c "ps -eo etime,cmd | grep -E 'extrair_|servico_t04' | grep -v grep"
```

"Em execução" significa vivo. "Pronto" significa terminado **ou morto** — e a
diferença só aparece no log e nos arquivos de saída.

O TCC 2 (documento de projeto) está em
`C:\Users\ferre\Downloads\TCC_2_Elian_Ferreira.pdf`.

---

## Documentos do projeto

| Arquivo | Conteúdo |
|---|---|
| `docs/AUDITORIA_TCC2_TCC3.md` | O que o TCC 2 pediu × o que foi entregue, lacuna a lacuna |
| `docs/RESULTADOS.md` | Todos os resultados experimentais e sua interpretação |
| `docs/DECISOES_METODOLOGICAS.md` | Divergências em relação ao projeto, com justificativa |
| `docs/T04_AMBIENTE_WSL2.md` | Os onze obstáculos vencidos para rodar os extratores de T04 |
| `docs/PLANO_ESCALA_INTEGRAL.md` | Viabilidade da escala e contingências |
| `docs/COMO_RODAR.md` | Como subir o sistema — VS Code e terminal —, a ordem obrigatória e o procedimento da defesa |
| `external/CONTRATO.md` | Integração com os repositórios oficiais |

Repositório: <https://github.com/elianferreira/tcc3-deteccao-imagens-ia> (privado)

---

## Trabalho futuro planejado

Registrado em 30/08/2026, a pedido. **Nada aqui foi executado.**

### 1. Campanha em escala integral

O treino atual usa **42.000 imagens** (split `train` de
`data/manifesto30k_standard.csv`, corpus de 60.000). A escala integral leva o
treino a **126.000** (`data/manifesto_escala_standard.csv`, corpus de 180.000
sobre as 192.638 montadas em disco).

Isto reverte a decisão de 16/08 de não executar a escala. Comando registrado na
seção "Decisões tomadas" deste arquivo.

### 2. Geradores de 2025/2026 — testar antes de treinar

Duas etapas, **nesta ordem**, e a ordem é o ponto:

1. **Testar primeiro.** Avaliar a arquitetura já treinada sobre imagens de
   geradores atuais, sem retreinar nada. Isso mede se ela **se mantém boa** em
   material que não existia quando o corpus foi montado — é uma medição de
   generalização temporal, e o resultado vale por si, positivo ou negativo.
2. **Treinar depois.** Só então incorporar esses geradores ao treino.

Inverter a ordem destruiria a medição: uma vez que os geradores novos entram no
treino, não há mais como saber como a arquitetura se comportava sem eles. É o
mesmo raciocínio que torna o protocolo OOD informativo.

O corpus de teste atual já cobre até Flux e Midjourney v6.1; o que falta são
geradores posteriores à montagem do corpus.

---

## 🔧 INTERFACE — teste de 31/08/2026 e o que falta consertar

Teste feito pelo Chrome DevTools sobre `localhost:7860`, com a imagem
`dalle3_00000` (sintética) em duas versões: nativa 1024x1024 e normalizada
256x256. Onze análises. Captura em `results/figuras/interface_dalle3_256px.png`.

### Achado 1 — as técnicas se invertem conforme a resolução do envio

| Técnica | 1024 px nativo | 256 px normalizado |
|---|---|---|
| T01 | **0,0%** errado | **99,5%** correto |
| T02 | **99,9%** correto | **0,0%** errado |
| T03 | 45,8% | 59,0% |
| T04 | 6,2% | 43,1% |
| **T05** | **38,2%** errado | **8,0%** errado |

A imagem é sintética nas duas. T01 e T02 **trocam de lado exatamente**.

> **Consertado em 31/08/2026** pelo item 1 abaixo. A coluna da esquerda deixou
> de existir: o envio de 1024 px agora é normalizado antes de analisar e produz
> exatamente a coluna da direita.

É a confirmação visual da seção 4.9: a T02 é espectral e precisa da resolução
nativa; a T01 foi treinada nas imagens normalizadas. **A T05 erra nas duas** —
é o item 7 do resumo do `RESULTADOS.md` (fusão de pesos fixos herda a
fragilidade dos componentes), agora visível na tela.

**Risco de demonstração:** enviar uma imagem de gerador moderno na frente da
banca faz o sistema responder "provavelmente real", em qualquer das duas
resoluções.

### Achado 2 — o RNF01 é cumprido, mas decai com a ociosidade

Quatro rodadas com o processo **recém-iniciado**, imagem de 256 px:

| Módulo | Média medida | Documentado (4.12.1) |
|---|---|---|
| T01 | 6,39 s | 4,56 s |
| T02 | 19,97 s | 13,89 s |
| T03 | 0,10 s | 0,08 s |
| T04 | 1,82 s | 1,50 s |
| **Total** | **21,6 s** | **20,03 s** |

**Passa** — 21,6 s contra o teto de 30 s, margem de 8,4 s. O número da 4.12.1
reproduz. Mas a mesma imagem, no processo com 14 h de ociosidade, custou
**54,2 s**; e em 1024 px, no processo frio, **214 s**.

| Estado do processo | Primeira análise, 256 px |
|---|---|
| Recém-iniciado | 22,1 s ✅ |
| 14 h ocioso | 54,2 s ❌ |

O `_aquecer()` da carga resolve o primeiro usuário depois que a interface sobe,
que é o que a 4.12.1 diz que ele resolve. **Não** resolve o processo que ficou
horas parado disputando 6 GB de VRAM com Teams, WhatsApp, Edge WebView e
Kaspersky. O mecanismo (despejo do contexto CUDA) é a explicação mais provável,
**não medida**.

> **Procedimento para a defesa: reiniciar a interface pouco antes de
> demonstrar.** Sem isso o requisito documentado como cumprido é violado na
> frente da banca.

### Achado 3 — a T04 é o módulo mais bem comportado

1,82 s de média contra 1,50 s documentado. O custo de tê-la posto na tela em
30/08 está confirmado como pequeno, independentemente do que for decidido na C3
sobre exibir ou não o escore.

---

## 🔬 VARREDURA POR PASTA — 31/08/2026

> ⛔ **PARCIALMENTE RETIRADA.** As pastas `fodb`, `imagenet`,
> `open_images` e `raise` não são imagens reais — vieram de
> `data/amostra`, o corpus sintético de teste. As 15 demais (COCO e os 14
> geradores) são válidas. Ver a seção **RETRATAÇÃO** antes de usar
> qualquer número desta seção.

3 imagens de cada uma das **19 pastas distintas** (5 fontes reais, 14
geradores), pelo caminho exato da interface: T01/T03/T04 sobre 256 px
normalizado, T02 sobre nativa com teto de 1.536 px, T05 sobre os escores de
256 px. Escores calculados em lote — já verificado como idêntico bit a bit ao
cálculo individual.

Dados: `results/varredura_pastas/varredura.csv` · script no scratchpad da sessão.

**Não existe pasta `stylegan` no corpus.** O único GAN é o `gigagan`.

### Robustez: zero erros

**Nenhuma técnica falhou em nenhuma das 57 imagens.** Cobertura de 57/57 nas
sete colunas. A tolerância a falhas (RN07) não foi exercitada porque não houve
o que tolerar.

### O quadro — mediana das 3 imagens, em % de síntese

Para `fake`, alto é certo; para `real`, baixo é certo. `*` marca erro.

| Pasta | T01 | T02 256px | **T02 nativa** | T03 | T04 | T05 4-fontes | T05 var. B |
|---|---|---|---|---|---|---|---|
| real/coco | 0,0 | 0,0 | 0,0 | 51,4* | 14,6 | 0,0 | 2,6 |
| ⛔ ~~real/fodb~~ (ruído) | **99,0\*** | 0,2 | 0,2 | 75,0* | 18,0 | 16,9 | 97,4* |
| ⛔ ~~real/imagenet~~ (ruído) | **89,9\*** | 0,0 | 0,0 | 72,2* | 17,5 | 8,1 | 93,5* |
| ⛔ ~~real/open_images~~ (ruído) | **98,8\*** | 0,2 | 0,2 | 75,0* | 17,5 | 15,4 | 96,4* |
| ⛔ ~~real/raise~~ (ruído) | **77,1\*** | 0,0 | 0,0 | 75,0* | 17,7 | 3,4 | 86,3* |
| fake/adobe_firefly | 4,7* | 0,0* | **100,0** | 27,4* | 25,8* | 0,0* | 0,9* |
| fake/dalle2 | 0,0* | 0,0* | **100,0** | 30,2* | 43,8* | 0,0* | 0,8* |
| fake/dalle3 | 0,2* | 0,0* | **100,0** | 57,6 | 29,9* | 0,0* | 3,6* |
| fake/flux | 60,1 | 0,2* | 35,4* | 45,6* | 26,0* | 0,5* | 35,6* |
| fake/gigagan | 7,0* | 55,3 | **100,0** | 42,8* | 45,4* | 33,0* | 67,2 |
| fake/glide | 100,0 | 88,9 | 88,9 | 49,4* | 18,9* | 99,5 | 99,8 |
| fake/latent_diffusion | 100,0 | 100,0 | 100,0 | 81,4 | 38,3* | 99,9 | 100,0 |
| fake/midjourney_v5 | 72,4 | 2,0* | **100,0** | 48,4* | 44,9* | 4,1* | 65,5 |
| fake/midjourney_v6_1 | 17,2* | 0,0* | 25,3* | 33,2* | 30,8* | 0,0* | 10,1* |
| fake/stable_diffusion_1_3 | 0,2* | 100,0 | 100,0 | 35,0* | 19,2* | 35,4* | 66,3 |
| fake/stable_diffusion_1_4 | 0,0* | 100,0 | 100,0 | 35,0* | 11,7* | 16,3* | 17,8* |
| fake/stable_diffusion_2 | 92,7 | 3,2* | **100,0** | 72,2 | 13,3* | 8,1* | 72,2 |
| fake/stable_diffusion_3 | 0,0* | 0,0* | 0,0* | 43,0* | 17,1* | 0,0* | 1,7* |
| fake/stable_diffusion_xl | 0,3* | 0,0* | **100,0** | 68,0 | 27,5* | 0,1* | 28,8* |

### Placar

> ⛔ A coluna **Reais** conta 5 pastas, das quais **4 são ruído**. Só a
> COCO é fonte real. A coluna Sintéticas permanece válida.

| Técnica | Pastas certas | Reais | Sintéticas |
|---|---|---|---|
| **T02 nativa** | **16/19** | 5/5 | **11/14** |
| T02 256 px | 10/19 | 5/5 | 5/14 |
| T05 quatro fontes | 7/19 | 5/5 | 2/14 |
| T05 variante B | 7/19 | 1/5 | 6/14 |
| T01 | 6/19 | **1/5** | 5/14 |
| T04 | 5/19 | 5/5 | **0/14** |
| T03 | 4/19 | **0/5** | 4/14 |

### ⛔ Achado 1 — RETIRADO em 31/08/2026

> ~~a T01 acusa de sintética toda fonte real que não seja a COCO~~
>
> **As quatro fontes são ruído procedural de `data/amostra`.** Um
> detector rejeitá-las não é viés — é acerto. O texto abaixo fica
> preservado como registro do raciocínio, não como resultado. Ver
> **RETRATAÇÃO**.

| Fonte real | T01 |
|---|---|
| coco | 0,0% ✓ |
| raise | **77,1%** ✗ |
| imagenet | **89,9%** ✗ |
| open_images | **98,8%** ✗ |
| fodb | **99,0%** ✗ |

**A única fonte real que a T01 classifica corretamente é a COCO — a única que
ela viu no treino.** Nas outras quatro ela acusa com 77 a 99% de confiança.

Isso é viés de conjunto de dados, exatamente o que a introdução do TCC 2
alertava: a rede aprendeu *"COCO versus latent_diffusion"*, não *"real versus
sintético"*. E tem consequência direta sobre número reportado: **a FPR de
0,0233 da T01 na tabela do protocolo padrão é medida só contra a COCO.** Contra
outras fontes reais ela seria de outra ordem.

Duas hipóteses concorriam:

1. **Viés de fonte** — a rede decorou as regularidades da COCO.
2. **Viés de formato** — a COCO é JPEG e carrega artefato de compressão que
   sobrevive à normalização; as quatro fontes da amostra são PNG nativo em
   256 px. A T01 teria aprendido "tem artefato de JPEG → real".

#### O teste que separou as duas — executado em 31/08/2026

20 imagens de cada fonte, comprimidas em JPEG a q95/85/75/50 sobre a imagem já
normalizada (é a ordem em que a COCO de fato passou: JPEG antes de virar PNG),
e reavaliadas. Dados em `results/teste_jpeg_t01/teste_jpeg.csv`.

O controle simétrico é o que dá força ao resultado: o mesmo tratamento no
`latent_diffusion`, que é **sintético** e que a T01 acerta com 100%. Se o
formato mandasse, comprimi-lo deveria empurrá-lo para "real" também.

**T01 — mediana do escore de síntese (%)**

| Pasta | PNG | q95 | q85 | q75 | q50 | Δ |
|---|---|---|---|---|---|---|
| fake/latent_diffusion | 100,0 | 65,1 | 84,4 | 52,6 | **0,0** | **−100,0** |
| real/coco | 0,0 | 0,0 | 0,1 | 0,0 | 0,0 | −0,0 |
| real/fodb | 91,9 | 97,6 | 92,9 | 91,1 | **84,2** | −7,6 |
| real/imagenet | 94,0 | 99,1 | 86,2 | 86,1 | **93,9** | −0,1 |
| real/open_images | 96,4 | 98,4 | 92,2 | 90,8 | **90,3** | −6,0 |
| real/raise | 71,6 | 98,8 | 86,6 | 91,5 | **72,7** | +1,1 |

#### ⛔ A hipótese de formato está REFUTADA — conclusão RETIRADA

> A refutação vale: a T01 **é** extremamente sensível a JPEG, e o colapso
> do `latent_diffusion` (dado real) prova isso. O que **não** se sustenta
> é o passo seguinte — "por eliminação, o viés é de fonte" —, porque as
> quatro fontes que sobreviveram à compressão são texturas sintéticas.
> Ver **RETRATAÇÃO**.

**As quatro fontes reais continuam acusadas em 72 a 94% mesmo em JPEG q50.**
Elas praticamente não se movem: −7,6, −0,1, −6,0 e +1,1 p.p. Se o mecanismo
fosse "artefato de JPEG → real", elas teriam desabado. Não desabaram.

E o controle simétrico fecha o argumento pelo outro lado: o `latent_diffusion`
**colapsa por completo**, de 100,0% para 0,0%. Ou seja, a T01 *é* extremamente
sensível a JPEG — mas essa sensibilidade não explica o erro nas fontes reais,
porque o que faz `fodb`, `imagenet`, `open_images` e `raise` parecerem
sintéticas **sobrevive à compressão que destrói a detecção do
`latent_diffusion`**. São dois fenômenos distintos.

~~**Por eliminação, o viés é de fonte.** A T01 aprendeu algo específico da
COCO — ou específico dessas quatro fontes — que não é artefato de
compressão.~~ ⛔ **RETIRADO:** as quatro fontes são ruído procedural. O
que sobreviveu à compressão foi a textura sintética, não uma regularidade
de fonte real.

O colapso sob JPEG, por sua vez, **não é novidade**: corrobora a seção 5.2 de
`RESULTADOS.md`, que já mede a T01 caindo de 0,9954 para 0,5842 em q50. O que
esta tabela acrescenta é a *direção* do erro — a compressão empurra a sintética
para "real", não para o acaso simétrico.

#### De quebra, a T03 ficou caracterizada com precisão

> ⛔ Só as linhas `real/coco` e `fake/latent_diffusion` são dado real. As
> quatro de −30 p.p. são ruído. O **+53,6 p.p. da COCO** é o dado que
> importa e permanece válido — a assinatura de dupla compressão.

| Pasta | PNG | q50 | Δ |
|---|---|---|---|
| real/coco | 17,2 | **70,8** | **+53,6** |
| real/fodb | 74,8 | 44,6 | −30,2 |
| real/imagenet | 73,7 | 43,9 | −29,8 |
| real/open_images | 74,6 | 44,0 | −30,6 |
| real/raise | 74,9 | 44,6 | −30,3 |
| fake/latent_diffusion | 84,9 | 69,2 | −15,7 |

A COCO vai na direção **oposta** às demais: +53,6 p.p. contra −30 p.p. É a
assinatura de **dupla compressão** — recomprimir o que já era JPEG produz um
padrão de coeficientes DCT distinto de comprimir um PNG nativo. Confirma por um
terceiro caminho o que as seções 2 e 5.3 já dizem: a T03 lê histórico de
compressão, não origem da imagem.

#### O que isto exige do texto

A FPR de 0,0233 da T01 e a de 0,0042 da T05 **são medidas contra a COCO
apenas** — isto continua verdadeiro, e continua pertencendo às ameaças à
validade. ⛔ Mas ~~contra outras fontes reais a T01 erra em 72 a 99%~~ está
**retirado**, e a medição proposta foi executada e **falhou no controle**:
não existe no corpus nenhuma fonte real além da COCO com que comparar.
Ver **RETRATAÇÃO**.

### Achado 2 — a T02 nativa é disparado a melhor técnica

16/19 pastas contra 10/19 em 256 px, e o ganho está todo nas sintéticas:
**5/14 → 11/14**. Sete geradores saem de 0,0% para 100,0% só por receber a
resolução nativa: adobe_firefly, dalle2, dalle3, midjourney_v5,
stable_diffusion_2, stable_diffusion_xl e gigagan.

Isso valida empiricamente a opção (b) da seção 1b, com margem muito maior do
que o caso único que motivou a mudança.

**Ressalva honesta:** as quatro fontes reais da amostra já estão gravadas em
256 px, de modo que para elas "nativa" e "256 px" são a mesma imagem — as
colunas são idênticas. O ganho medido está inteiramente nas sintéticas.

Três geradores resistem mesmo em nativa: **stable_diffusion_3 (0,0%)**,
**midjourney_v6_1 (25,3%)** e **flux (35,4%)** — os três mais recentes do
corpus. Coerente com a seção 3.3: o padrão temporal continua valendo.

### Achado 3 — a T03 não acerta nenhuma pasta real

0/5 nas reais, com 51 a 75% em todas as cinco, e 4/14 nas sintéticas. Como está
na interface, ela contribui ruído com aparência de informação. Já se sabia que
ela media compressão e não síntese (seção 2 de `RESULTADOS.md`); a varredura
mostra o efeito na tela.

### Achado 4 — a T04 responde baixo para tudo

5/5 nas reais e **0/14 nas sintéticas**, com escores entre 11,7% e 45,4% em
todas as 19 pastas. Ela "acerta" as reais por sempre responder baixo, não por
discriminar. É a AUC 0,53 vista de perto, e confirma que a caixa dela na tela
exibe ruído — o que já estava registrado como custo aceito de demonstrabilidade.

### Achado 5 — nenhuma das duas fusões é utilizável

As duas erram, e em direções opostas:

| | Reais | Sintéticas |
|---|---|---|
| T05 quatro fontes | 5/5 | **2/14** — quase nunca detecta |
| T05 variante B | **1/5** | 6/14 — detecta mais, e acusa 4 fontes reais |

A de quatro fontes é conservadora ao ponto da inutilidade; a variante B herda o
viés da T01 e acusa `fodb`, `imagenet`, `open_images` e `raise`. **A escolha
entre elas, discutida na seção 2b, é entre dois modos de errar** — o que reforça
que a recalibração de lá não basta sozinha se a T01 não for investigada.

### Achado 6 — confundidores de formato no corpus nativo

| Pasta | Extensão | Conteúdo real |
|---|---|---|
| fake/adobe_firefly | `.png` | **JPEG** |
| fake/stable_diffusion_3 | `.webp` | WEBP |
| real/coco | `.jpg` | JPEG |
| demais | `.png` | PNG |

**O `adobe_firefly` tem extensão `.png` com conteúdo JPEG** — divergência entre
extensão e formato dentro do corpus nativo. A normalização converte tudo para
PNG e neutraliza isso no corpus usado nas medições, mas quem enviar o arquivo
nativo à interface passa pelo `validate_upload`, que confere o conteúdo e não a
extensão — então aceita, corretamente. Vale registrar porque alimenta a
hipótese de formato do Achado 1.

### O teto de 1.536 px quase não é acionado

3 de 57 imagens — todas do `adobe_firefly` (2688x1536 → 1536x878). Os demais
geradores ficam em 1.024 a 1.344 px e passam intactos. O teto protege o caso
extremo sem tocar no resto.

## O QUE FALTA CONSERTAR NA INTERFACE — em ordem de retorno

Item 1 feito em 31/08/2026. **O próximo é o item 2**, que é o de maior ganho:
a T02 responde por cerca de 39 s dos 52 s medidos na tela.

### 1. Normalizar a imagem enviada  ·  ✅ FEITO em 31/08/2026

`app/gradio_app.py` mandava o upload **cru** para as cinco técnicas, enquanto
todos os números da dissertação foram medidos sobre imagens normalizadas. Era a
causa direta do Achado 1.

**O que foi feito.** Um gerenciador de contexto `envio_normalizado()`
(`app/gradio_app.py:287`) aplica o `resize_and_center_crop()` de
`scripts/normalize_corpus.py` ao upload e entrega o caminho normalizado tanto
para `service.analyze()` quanto para o espectro exibido — as duas saídas
precisam ver a mesma imagem. Antes, o espectro ainda espremia uma 1024x768 num
quadrado de 256, distorcendo justamente a geometria que se pede ao usuário
para ler.

**Prova de que é a condição medida.** O arquivo entregue às técnicas a partir
do envio nativo de `dalle3_00000` (1024x1024, RGBA) é **pixel a pixel idêntico**
a `data/corvi2024_30k_norm/fake/dalle3/dalle3_00000.png`, isto é, ao arquivo do
corpus sobre o qual o Capítulo 4 foi medido.

**Prova na tela.** Enviando o arquivo **nativo de 1024 px**, a interface agora
reproduz exatamente a coluna "256 px normalizado" do Achado 1. Captura em
`results/figuras/interface_dalle3_1024_normalizado_resultado.png`.

| Técnica | 1024 px, antes | 1024 px, agora | 256 px do Achado 1 |
|---|---|---|---|
| T01 | 0,0% errado | **99,5% correto** | 99,5% |
| T02 | 99,9% correto | 0,0% errado | 0,0% |
| T03 | 45,8% | 59,0% | 59,0% |
| T04 | 6,2% | 43,1% | 43,1% |
| T05 | 38,2% errado | 8,0% errado | 8,0% |

Efeito colateral honesto, como previsto: a T02 fica pior na tela. Não é dano do
conserto — é o comportamento medido dela neste corpus, os -17,8 p.p. da 4.9.

**Custo medido.** 30 ms num envio de 1024 px e 240 ms numa 2304x1792. Uma imagem
já conforme — 256 x 256 em RGB, que é o corpus inteiro — é devolvida **sem
cópia**, a 1,0 ms, para que `scripts/medir_rnf01_com_t04.py` siga medindo o
mesmo caminho de antes.

**O conserto não custa tempo, e há controle para isso.** As três execuções pela
interface viva ficaram em 48–54 s de total, muito acima dos 21,6 s do Achado 2.
Não é o conserto: o **controle** — enviar o arquivo já normalizado de 256 px,
que o código devolve sem tocar, percorrendo exatamente o caminho de antes da
mudança — deu 52,4 s, dentro da mesma faixa.

| Envio | T01 | T02 | T03 | T04 | Total |
|---|---|---|---|---|---|
| 1024 px, normalizado pelo conserto | 11,40 | 39,89 | 0,33 | 2,10 | 53,7 s |
| 1024 px, terceira rodada | 10,78 | 35,56 | 0,29 | 1,81 | 48,4 s |
| **256 px já conforme (controle)** | 11,53 | 38,90 | 0,17 | 1,78 | **52,4 s** |

O excesso é o Achado 2 outra vez, agravado por o Chromium do Playwright estar
disputando a máquina durante a medição — e a T02 responde por 39 s dos 52 s.
**É exatamente o que o item 2 conserta.** Para reproduzir os 21,6 s vale o
procedimento da defesa: reiniciar a interface e medir sem navegador competindo.

**Regressão coberta:** oito testes novos em `tests/test_interface.py` (doze
casos, com a parametrização de resoluções) fixam
256x256 em RGB para envios de 1024x1024, 1024x768, 300x900 e 64x64; a igualdade
com `resize_and_center_crop`; a devolução sem cópia do já conforme; a conversão
do 256 px em escala de cinza; a remoção do temporário no caminho normal e sob
exceção (RNF03); e o fechamento do descritor da origem, de que a T02 depende no
Windows. Suíte completa: **137 passam**.

### 1b. A resolução da T02 — medido em 31/08/2026

O item 1 pôs T01, T03 e T04 na condição medida e, ao fazê-lo, **empurrou a T02
para a pior condição dela**. Como a T02 é a fonte de maior peso da fusão, levou
a T05 junto. Esta seção mede o problema e registra a saída escolhida.

#### A T02 satura — 0,0% é a resposta padrão dela, não um veredito

48 imagens normalizadas a 256 px, seis geradores modernos mais reais do COCO:

| Grupo | n | mediana | exibidas como 0,0% |
|---|---|---|---|
| dalle3 | 6 | 0,0000 | 100% |
| stable_diffusion_3 | 6 | 0,0000 | 100% |
| midjourney_v6_1 | 6 | 0,0000 | 83,3% |
| adobe_firefly | 6 | 0,0000 | 83,3% |
| flux | 6 | 0,0000 | 66,7% |
| midjourney_v5 | 6 | 0,0102 | 50,0% |
| **REAIS (coco)** | 12 | 0,0000 | **91,7%** |

**A T02 a 256 px responde 0,0% para 91,7% das reais e 80,6% das sintéticas.**
Nessa resolução o número quase não carrega informação — é a resposta padrão.

A saída da SPAI é saturada, praticamente binária: no split de teste padrão as
reais têm mediana 0,000000 (87,9% são exatamente zero) e as sintéticas mediana
1,000000. Quando ela erra, erra com 0,0% na tela, não com um 40% hesitante.

**A AUC 0,9976 da T02 não contradiz isso.** O único gerador sintético do split
de teste padrão é `latent_diffusion`, 4.500 imagens, nativo 256 px — condição
que a normalização não prejudica. Aquele número não é evidência de que a T02
funcione em geradores modernos a 256 px.

#### A T02 domina a fusão

Pesos de `t05_fusion__quatro_fontes.pkl`, sobre escores padronizados:

| Fonte | Peso |
|---|---|
| T01 | +3,43 |
| **T02** | **+4,27** ← o maior |
| T03 | +0,52 |
| T04 | −0,25 |

Contrafactuais a partir de um caso real (T01 97,7%, T02 0,0%, T03 30,6%,
T04 69,7% → T05 2,5%):

| Mexendo só na T02 | T05 | | Mexendo só na T01, com T02=0 | T05 |
|---|---|---|---|---|
| T02 = 0,00 | 2,5% | | T01 = 0,000 | 0,0% |
| T02 = 0,50 | 66,2% | | T01 = 0,500 | 0,1% |
| T02 = 1,00 | 99,3% | | T01 = 1,000 | 3,0% |

**Com a T02 zerada, a T05 é praticamente um repetidor da T02.** A T01 pode ir de
0% a 100% que a T05 só sai de 0,0% para 3,0%. É o item 7 do resumo com número.

#### O teste que decide: nenhuma resolução única funciona

`adobe_firefly_00002`, nativa 2688x1536 (4,13 MPx), **sintética**:

| Técnica | Nativa | Normalizada 256 px |
|---|---|---|
| T01 | 2,5% ✗ | **97,7%** ✓ |
| T02 | **99,8%** ✓ | 0,0% ✗ |
| T03 | 41,6% | 30,6% |
| T04 | 69,8% ✓ | 69,7% ✓ |

| Estratégia | Fontes que acertam | T05 |
|---|---|---|
| A) tudo normalizado | 2/4 | 2,5% ✗ |
| B) tudo nativo | 2/4 | 19,5% ✗ |
| **C) misto: T02 nativa, resto normalizado** | **3/4** | **99,3%** ✓ |

**As duas estratégias de resolução única erram.** A mista é a única que põe T01
e T02 do mesmo lado. E a política por técnica não é chute: a **T04 dá o mesmo
escore nas duas** (69,8 × 69,7) e custa **35,3 s nativa contra 3,0 s
normalizada** — 12× mais cara por nada.

#### O obstáculo: a T02 nativa consome a placa inteira

Memória amostrada durante a chamada nativa, com o Gradio **parado**:

| | Pico |
|---|---|
| GPU | **5.931 MiB de 6.144** — 96,5% da placa |
| RAM livre | caiu a **1.038 MB** de 16 GB |

É o que o `normalize_corpus.py` já registrava como tendo "inviabilizado a
inferência na GPU de 6 GB". Na primeira tentativa, com o Gradio no ar, **a
memória esgotou e derrubou os dois serviços T04 do WSL2** — foi preciso
relançá-los. Por isso a resolução entregue à T02 **precisa de teto**; a RN02
aceita 10 MB, que podem ser bem maiores que 4 MPx.

#### A decisão: opção (b)

Três saídas foram consideradas para o conflito entre exibir a T02 no seu melhor
e preservar a configuração em que a T05 foi medida:

- **(a) Declarar** — a T02 recebe nativa e a T05 na tela vai rotulada como fora
  da configuração medida. Barato, mas o número principal deixa de ser o do
  capítulo.
- **(b) Dois escores de T02** — o de 256 px alimenta a T05, preservando a
  configuração medida; o de resolução nativa aparece na tela como a resposta da
  própria técnica. **Escolhida em 31/08/2026.**
- **(c) Reajustar a fusão** com T02 em nativa sobre o corpus — o mais limpo e o
  mais caro; exige a passada completa que o hardware inviabilizou. Decisão de
  pesquisa, pertence à A3.

A (b) preserva a armadilha nº 6: a fusão continua recebendo escores da mesma
distribuição em que foi ajustada. O custo é uma segunda passada de T02 por
imagem — o que torna o **item 2 pré-requisito**, e não mera otimização.

**Isto deixou de ser conserto de interface e virou escolha metodológica.** Vale
entrar na pauta perto da A3 e do bloco C: é a tensão da 4.9 aparecendo na tela.

#### A opção (b) implementada — 31/08/2026

`envio_para_t02()` (`app/gradio_app.py`) é um segundo gerenciador de contexto,
independente do `envio_normalizado()`, para não invalidar os testes do item 1.
A T02 passa a ser avaliada **duas vezes** por análise: o escore de resolução
nativa vai para a tela, o de 256 px continua alimentando a T05.

| Chave | Valor |
|---|---|
| `TCC3_T02_NATIVA` | `1` (padrão). `0` devolve o comportamento anterior |
| `TCC3_T02_TETO` | `1536` px no maior lado |

**O teto não é cosmético — é o que torna a coisa operável.** Com o teto, o pico
de GPU na análise foi **3.108 MiB**; sem teto, na resolução nativa de 4,13 MPx,
foram **5.931 MiB de 6.144**. E os serviços T04 do WSL2 **sobreviveram** à
análise com teto, depois de terem sido derrubados duas vezes pelas chamadas sem
teto.

O teto de 1.536 px vem da leitura da 4.9 (redução de até 2× fica na tolerância),
**não de medição própria**. A medição de ~200 imagens que o fixaria continua em
aberto.

##### O que a tela mostra agora

`adobe_firefly_00002` nativa, com as quatro fontes disponíveis:

| Técnica | Antes da (b) | Depois da (b) |
|---|---|---|
| T01 | 97,7% ✓ | 97,7% ✓ |
| **T02** | **0,0% ✗** | **100,0% ✓** · resolução nativa |
| T03 | 30,6% | 30,6% |
| T04 | 69,7% ✓ | 69,7% ✓ |
| **T05** | **2,5% ✗** | **2,5% ✗** |

##### A (b) conserta a T02 e NÃO conserta a T05 — e isso é por desenho

Vale registrar sem suavizar, porque contraria a expectativa com que a opção foi
escolhida. O teste de bancada mostrou T05 = 99,3% na estratégia mista; **aquele
número era da estratégia (c)**, que alimenta a fusão com o escore nativo. A (b)
faz o oposto por definição: preserva a distribuição de ajuste da fusão, logo a
T05 continua comendo o 0,0% de 256 px e continua respondendo 2,5%.

Consequências, todas verdadeiras ao mesmo tempo:

- A T02 deixou de mentir na tela. Ela agora mostra do que é capaz — 100,0%.
- A T05 continua errando, e o **risco de demonstração do Achado 1 permanece**:
  o número de maior destaque ainda diz "provavelmente real" para um gerador
  moderno.
- Em compensação, a **discordância ficou legível** — três fontes acima de 69% e
  a fusão em 2,5%, com o rodapé explicando por quê. É exatamente a melhoria que
  a seção "O que NÃO tem conserto" apontava como possível.

Quem quiser a T05 correta na tela precisa da **opção (c)** — reajustar a fusão
com escores de T02 em resolução nativa sobre o corpus. É decisão de pesquisa e
pertence à A3, não ao conserto de interface.

##### RNF01 continua violado

T01 5,99 + T02 37,87 + T03 0,12 + T04 2,10 = **46,1 s**, contra o teto de 30 s.
A T02 responde por 37,9 s — agora com duas passadas. **O item 2 deixou de ser
otimização e virou pré-requisito**: sem a T02 residente, a (b) não cabe no
RNF01.

##### O lote misto — uma chamada em vez de duas

A primeira implementação da (b) chamava a T02 duas vezes, pagando o arranque do
subprocesso em dobro. `predict_proba` **já opera em lote**: escreve todos os
caminhos num CSV e lança um único subprocesso. Uma chamada com os dois caminhos
resolve.

Medido em 31/08/2026, fora da interface:

| | Escore 256 px | Escore nativo | Tempo |
|---|---|---|---|
| Duas chamadas | 9,584388e-05 | 0,99983263 | 51,16 s |
| **Uma chamada** | 9,584388e-05 | 0,99983263 | **21,49 s** |

**Idênticos bit a bit** — a diferença é 0,000e+00 nos dois escores. A
preocupação legítima (misturar 256×256 com 1536×878 no mesmo lote poderia
perturbar o resultado, como o lote 1 × lote 128 da T04 perturbou em 2,3e-04)
**não se materializou**.

Na interface o ganho é menor que fora dela, porque o modelo já está em cache do
sistema:

| | Antes do lote | Depois |
|---|---|---|
| T02 na tela | 37,87 s | **30,02 s** |
| Total da análise | 46,1 s | **41,8 s** |

**Continua violando o RNF01.** A projeção de ~28 s que motivou o atalho era
otimista: ela usava a economia medida fora da interface. O atalho desfaz o
desperdício introduzido pela (b), mas não ataca os ~20 s de arranque —
**só o item 2 ataca**.

Há um resguardo: se a chamada conjunta falhar (o modo esperado é falta de
memória na imagem grande), o código refaz a passada só com os 256 px, que é o
que a fusão precisa. Juntar as duas passadas não pode custar o escore da fusão.

##### Armadilha nova, custou duas quedas

**A T02 em resolução nativa sem teto derruba os serviços T04 do WSL2** por
esgotamento de VRAM, e a queda é silenciosa — as portas 8404/8405 simplesmente
param de responder. Aconteceu duas vezes em 31/08. Pior: a `is_available()` da
T04 roda **na carga da interface**, então subir o Gradio antes dos serviços
marca a T04 como indisponível pelo resto da sessão. **A ordem é: serviços T04
primeiro, interface depois.**

### 2. T02 residente, no molde do serviço da T04  ·  meia tarde  ·  maior ganho

`src/techniques/t02_spai.py:173` abre um **subprocesso por chamada**, num venv
separado. Cada análise paga arranque de interpretador e carga de modelo: ~20 s
na tela contra 555 ms/imagem em lote.

Conserto: processo residente que carrega o SPAI uma vez e responde por
requisição. O molde inteiro já existe — `scripts/wsl/servico_t04.py`,
`scripts/wsl/servico_t04_controle.sh` e, principalmente,
`scripts/verificar_paridade_servico_t04.py`, que **prova que os escores não
mudam**. Sem essa prova de paridade a mudança não se sustenta.

Efeito: total da interface de 21,6 s para cerca de 8 s, e a decadência por
ociosidade encolhe junto, porque o custo dominante some.

### 2b. A interface trocou de modelo de fusão sem ninguém decidir  ·  ACHADO

Descoberto em 31/08/2026 ao investigar por que a T02 domina a T05.

**Ninguém decidiu que a T02 prevaleceria.** O peso é *aprendido*, e depende
inteiramente de em que partição a fusão foi calibrada. Os três modelos em
`weights/`:

| Modelo | T01 | T02 | T03 | T04 | Maior peso |
|---|---|---|---|---|---|
| `t05_fusion.pkl` (variante B) | **+2,671** | +2,027 | +1,146 | — | **T01** |
| `t05_fusion__variante_A.pkl` | +3,495 | **+3,735** | +1,075 | — | T02 |
| `t05_fusion__quatro_fontes.pkl` | +3,430 | **+4,266** | +0,520 | −0,251 | T02 |

E o efeito disso na tela, no mesmo `adobe_firefly_00002` (sintética):

| Modelo carregado | T05 |
|---|---|
| `t05_fusion.pkl` (variante B) | **67,7%** ✓ |
| `t05_fusion__quatro_fontes.pkl` | **2,5%** ✗ |

#### A troca é silenciosa e depende da T04

`DetectionService._load_all()` escolhe o modelo pela disponibilidade da T04:
com T04 no ar carrega o de quatro fontes; sem ela, o de três. Isso foi
observado por acidente nas capturas de hoje — com os serviços T04 caídos a tela
deu **67,7%**, e com eles no ar deu **2,5%**, sobre a mesma imagem.

A escolha em si é correta e está justificada no código: carregar o de três
fontes com a T04 disponível faria a fusão descartar a quarta entrada em
silêncio. **O problema é outro:** os dois modelos não foram calibrados na mesma
partição.

- `t05_fusion.pkl` é a **variante B**, calibrada sobre gerador *held-out*
  (`glide`) — a que a seção 4 de `RESULTADOS.md` mostra operar num ponto
  utilizável.
- `t05_fusion__quatro_fontes.pkl` saiu de `scripts/pipeline_t04_fusao.py`, que
  roda `run_experiments.py --protocol standard --refit-fusion` **sem**
  `--fusion-split fusion`. O padrão de `--fusion-split` é `val`, que o próprio
  `run_experiments.py:199` descreve como *"reproduz a calibracao
  in-distribution original"*.

Ou seja, o modelo que a interface usa hoje foi calibrado no regime que a seção
"Por que isso não funcionou" de `DECISOES_METODOLOGICAS.md` identifica como o
que quebra: *"a fusão foi calibrada num cenário fácil e aplicada num cenário
difícil, sem nunca ter visto como suas fontes se comportam quando erram"*. E no
protocolo padrão o cenário é ainda mais fácil que o da variante A — a T02 mede
AUC 0,9976 ali, a melhor das quatro, e a regressão lhe dá o maior peso.

**A T02 prevalecer é artefato do regime de calibração do modelo de quatro
fontes, não decisão de projeto.** É reversível.

#### A documentação ficou desatualizada

`RESULTADOS.md:1204` e `:1212` afirmam que *"a aplicação usa a variante B"*.
Isso deixou de ser verdade em 30/08, quando a T04 entrou na interface. As duas
linhas precisam de correção.

#### A possibilidade: reajustar o de quatro fontes na calibração da variante B

> ✅ **EXECUTADO em 31/08/2026, tarde.** Funcionou, e trouxe uma escolha
> que pertence à A3 — ver a seção **RECALIBRAÇÃO DA T05**. O texto abaixo
> fica como o levantamento que a viabilizou.

**Não executado.** Registrado em 31/08/2026 com o levantamento já feito, para
que a decisão não precise recomeçar do zero.

##### Por que é promissora

Este é o caminho mais direto para a T05 voltar a acertar na tela, e **não passa
pela questão de resolução**. Alimentando a variante B com o caso real
`adobe_firefly_00002` — inclusive com a T02 nos mesmos 0,0001 de 256 px — ela
responde **67,7%**, porque nela quem pesa mais é a T01 (+2,671), não a T02. Ou
seja: pode não ser preciso mexer em resolução nenhuma para a fusão parar de
errar.

##### O bloqueio, que é concreto

**`--fusion-split fusion` não roda no protocolo padrão.** Verificado:

| Manifesto | Partições |
|---|---|
| `manifesto30k_standard.csv` | val 9.000 · train 42.000 · test 9.000 — **sem `fusion`** |
| `manifesto30k_ood.csv` | val 9.000 · train 46.500 · test 14.788 · **fusion 2.350** |

A partição `fusion` só existe no OOD, e é `glide` (1.000) + reais (1.350).
Como `scripts/pipeline_t04_fusao.py` roda `--protocol standard`, o argumento
simplesmente não tem onde se aplicar. O reajuste teria de ser no **protocolo
OOD** — que é, aliás, onde a variante B foi ajustada.

E aí aparece o bloqueio de verdade: **não há escores de T04 para o `glide`.**
`results/t04_escores_combined.csv` tem 59.999 linhas e cobre apenas
`latent_diffusion` (30.000) e `coco` (29.999). Da partição `fusion`, só as
1.350 reais têm escore — **cobertura de 57,4%, e zero nas 1.000 sintéticas**.

##### O que seria preciso, na ordem

1. **Extrair T04 para as 1.000 imagens de `glide`**, com os três extratores.
   Custo estimado a partir da rodada do benchmark (14.788 imagens em 45 min de
   GPU + 203 + 111 min de CPU): **~25 a 30 min de máquina**. É barato.
2. Consolidar num arquivo de escores que cubra `latent_diffusion`, `coco` **e**
   `glide`.
3. Reajustar no OOD com a calibração da variante B:
   `run_experiments.py --protocol ood --refit-fusion --fusion-split fusion`,
   com `TCC3_T04_ESCORES` apontando para o arquivo do passo 2.

**Respeitar as armadilhas de sobrescrita** (nºs 1 a 4 desta seção) e a nº 5: ao
mudar a origem de um escore, apagar `results/scores/<protocolo>__<tecnica>__*`,
senão a campanha reaproveita o `.npy` antigo e reporta o valor errado em
silêncio.

##### Uma camada a mais do mesmo problema

A interface hoje **mistura protocolos** conforme a T04 esteja no ar:
`t05_fusion.pkl` (variante B) é do protocolo **OOD**; o de quatro fontes é do
protocolo **padrão**. Não é só calibração diferente — é protocolo diferente. Se
o reajuste for feito, os dois passam a ser OOD e essa inconsistência some junto.

##### O que isto NÃO resolve

A questão de resolução da seção 1b é independente. Mesmo com a fusão
reajustada, a T02 continua medindo 0,0% em 256 px sobre geradores modernos — o
que muda é a fusão parar de depender tanto dela. As opções (b) e (c) de lá
seguem valendo por conta própria.

##### Onde isto pertence

Provavelmente na **A3** da pauta, junto com a opção (c) da seção 1b: as duas
mexem em como a contribuição da T05 é constituída e reportada, e nenhuma é
conserto de interface.

### 3. Aquecimento periódico  ·  quinze minutos

`_aquecer()` (`app/gradio_app.py:64`) já existe e funciona; só roda uma vez, na
carga. Chamá-lo a cada 5 min numa thread de fundo mantém o contexto CUDA vivo.
Custo: uma inferência de 256x256 a cada 5 min.

### 4. Investigar a T01  ·  sem explicação ainda

**6,4 s por imagem na interface contra 5,8 ms em lote** — mil vezes mais lenta,
e ela é carregada em processo, sem subprocesso. Pode ser transferência CPU/GPU
por chamada, ou o cálculo do espectro entrando na conta. Segundo maior custo da
tela depois da T02.

### O que NÃO tem conserto, e não deveria ter

A T05 errar nas duas resoluções não é defeito — é achado, e já está no item 7
do resumo. Corrigir exigiria **retreinar a fusão**, que é decisão de pesquisa e
provavelmente pertence ao trabalho futuro, não ao TCC 3. O que a interface já
faz certo é não emitir veredito binário (RN04). O que dá para melhorar é tornar
a **discordância** legível na tela, em vez de deixar o usuário somar de cabeça.

---

## PONTO DE RETOMADA

Atualizado em 02/09/2026.

### O que está vivo agora

| | |
|---|---|
| Serviços T04 (portas 8404 e 8405) | **no ar**, há mais de dois dias |
| Interface (porta 7860) | **parada, de propósito** — derrubada para o Elian subir pelo VS Code |
| Rodadas em execução | **nenhuma** |

Para subir a interface: `docs/COMO_RODAR.md`. **Os serviços T04 já estão no ar,
então a ordem está satisfeita** — pode subir a interface direto.

---

### O que a sessão de 31/08 e 02/09 entregou

Duas frentes, na ordem em que aconteceram.

#### 31/08, manhã — interface

| | O quê | Onde |
|---|---|---|
| ✅ | **Item 1** — normalização do envio; a tela passou a corresponder ao capítulo | seção "1." |
| ✅ | **Item 1b** — T02 em resolução nativa com teto de 1.536 px (opção b) | seção "1b" |
| ✅ | **Atalho do lote** — uma chamada de T02 em vez de duas, idêntica bit a bit | seção "1b" |
| 📌 | **Achado 2b** — a interface troca de modelo de fusão sem ninguém decidir | seção "2b" |

#### 31/08, tarde — a retratação e a recalibração

| | O quê | Onde |
|---|---|---|
| ⛔ | **Varredura das 19 pastas** — 15 válidas, **4 retiradas** | "RETRATAÇÃO" |
| ⛔ | **Teste de JPEG** — a refutação vale, a conclusão de viés **caiu** | "RETRATAÇÃO" |
| 🔬 | **Controle de reprodução da T01** — FPR 0,0233 reproduz na 4ª casa | "RETRATAÇÃO" |
| 🔒 | **Travas contra o corpus sintético** — marcador + recusa no script | "RETRATAÇÃO" |
| ✅ | **Recalibração da T05** na partição `fusion` — 3/15 → 7/15 pastas | "RECALIBRAÇÃO DA T05" |
| 🔬 | **Controle de reprodução da T04** — 1.309/1.350 idênticas bit a bit | "RECALIBRAÇÃO DA T05" |
| ⚠️ | **A A3 virou escolha excludente** — os dois candidatos se anulam | "RECALIBRAÇÃO DA T05" |

#### 02/09 — ambiente de trabalho

| | O quê | Onde |
|---|---|---|
| ✅ | **VS Code 1.135.0 instalado e configurado**, com as quatro extensões | `docs/COMO_RODAR.md` |
| ✅ | **`.vscode/` versionado** — 9 configurações de execução e 6 tarefas | `docs/COMO_RODAR.md` |
| 📄 | **Runbook único** de como subir o sistema, pelos dois caminhos | `docs/COMO_RODAR.md` |
| ⛔ | **Não dá para iniciar a depuração de fora** — `SendKeys` não dispara o `F5` | `docs/COMO_RODAR.md` |

**148 testes passam** ao fim de tudo.

### Código e dados novos

| Arquivo | O quê |
|---|---|
| `scripts/medir_fpr_t01_fontes.py` | FPR por fonte real, com IC de Wilson; recusa corpus sintético |
| `scripts/t04_glide_fusao.bat` | extração de T04 na partição `fusion` do OOD |
| `scripts/recalibrar_t05_fusion.py` | funde escores de T04 e reajusta a T05, em ambiente isolado |
| `scripts/comparar_modelos_t05.py` | confronta os modelos de fusão sobre a mesma matriz |
| `.vscode/` | configuração de execução, tarefas e o script de subida dos serviços |
| `results/fpr_t01_fontes/` | o controle de reprodução da T01 |
| `results/t04_escores_componentes_ood_fusion.csv` | as três representações da partição `fusion` |
| `results/t04_escores_componentes_ood_completo.csv` | o arquivo fundido que cobre o OOD inteiro |
| `weights/t05_fusion__quatro_fontes_ood.pkl` | **o modelo recalibrado** — a interface ainda NÃO o usa |

---

### ⚠️ Nada foi commitado desde `42aae24`

São **41 arquivos** em três sessões. Isso deixou de ser detalhe:

- a armadilha nº 4 já destruiu números da dissertação uma vez, e eles só voltaram
  porque `git show HEAD:` os tinha;
- três dos quatro modelos de fusão em `weights/` são versionados, e o novo entrou
  hoje sem commit.

**Commitar é o próximo passo mais barato e o de maior proteção.**

---

### As frentes abertas, em ordem

**1. A A3 ficou com número dos dois lados, e virou escolha excludente.**
É a pergunta mais madura da pauta agora. A recalibração funciona na tela
(3/15 → 7/15) mas custa a FPR de 0,0044, que é justamente um dos candidatos a
contribuição da T05. **Não dá para ter os dois.** Ver "RECALIBRAÇÃO DA T05".

**2. Escrever o parágrafo de ameaças à validade.** Não depende de medição nem do
orientador: o corpus tem **uma única fonte real**, verificável em uma linha
(`{'coco': 30000}`), e por isso a FPR de 0,0233 da T01 e a de 0,0042 da T05 são
medidas contra a COCO e só. É a peça que hoje não existe em lugar nenhum.

**3. Redigir o Capítulo 4** — bloqueado por A1, A2 e A3. A A1 tem resposta
medida; a A3 agora tem número dos dois lados.

**4. Registrar no `RESULTADOS.md`** o que só existe aqui: a rodada da T04 sobre o
benchmark (seria a 4.14), os achados da interface, a retratação, a recalibração —
e **a correção das linhas 1204 e 1212**, que afirmam que "a aplicação usa a
variante B". É afirmação verificável e falsa desde 30/08, e é a mais urgente.

### Investigação encerrada por falta de dado

**O viés de fonte da T01.** A medição rodou, falhou no controle, e a evidência que
a motivava não existia. Medir de verdade exigiria **baixar imagens reais** de
FODB, ImageNet, Open Images ou RAISE — trabalho de corpus, não de análise. O
`medir_fpr_t01_fontes.py` as aceita sem alteração no dia em que existirem. É
pergunta de escopo para o orientador, bloco B.

### Melhorias de interface ainda abertas

| Item | O quê | Custo |
|---|---|---|
| 2 | T02 residente — único caminho para o RNF01 (hoje 41,8 s contra 30 s) | meia tarde + prova de paridade |
| ~~2b~~ | ✅ **Recalibração feita** — o modelo existe; **trocar o que a interface carrega é decisão da A3** | feito |
| 3 | Aquecimento periódico — mata a decadência por ociosidade | quinze minutos |
| 4 | Investigar a lentidão da T01 (6,4 s na tela × 5,8 ms em lote) | sem estimativa |
| — | Medir o teto de resolução da T02 (~200 imagens); hoje 1.536 px é inferido | curto |
| — | Avaliar o modelo recalibrado no split `test` do OOD | ~2 h (armadilha 7) |

## Onde o trabalho chegou

### As quatro técnicas e a quinta

| Técnica | Protocolo padrão | Situação |
|---|---|---|
| T01 — Coocorrência + CNN | AUC 0,9962 | replicada, multi-semente 0,9964 ± 0,0002 |
| T02 — SPAI | AUC 0,9976 | replicada e **verificada** contra o publicado |
| T03 — Benford/DCT | AUC 0,8064 | replicada |
| T04 — Geometria projetiva | AUC 0,5333 | **três representações extraídas**; resultado negativo |
| **T05 — Fusão (contribuição)** | **AUC 0,9996** | medida com as quatro fontes completas |

### O que T04 estabeleceu

Os três extratores foram postos de pé no WSL2 — SSISv2, PerspectiveFields e
DeepLSD —, o que exigiu vencer **onze obstáculos** não documentados
(`T04_AMBIENTE_WSL2.md`). As três representações medem em acaso neste corpus:

| Representação | AUC |
|---|---|
| Objeto-sombra | 0,5384 |
| Campos de perspectiva | 0,5355 |
| Segmentos de reta | 0,5187 |
| Média das três | 0,5333 |

Os mesmos classificadores, sobre os dados dos autores, dão 0,82. A replicação
está correta; o que não transfere é o método. E a média **não** supera a melhor
representação — elas erram junto, não de formas complementares.

Na fusão, T04 é indiferente: com e sem ela T05 fica em 0,9996, e o teste de
DeLong dá **p = 0,182** com os dois modelos ajustados na mesma validação. Isso
confirma por medição a tolerância a fonte inútil prevista em RN07 e RNF04.

Detalhes em `RESULTADOS.md`, seções 4.7 a 4.12.

---

## Concluído

- Repositório publicado no GitHub
- Análise de erros e Grad-CAM de T01 (Etapa 5)
- Corpus em escala montado e regenerado sem perfil ICC: 192.638 imagens
- Os três manifestos da escala (padrão, OOD, OOD por famílias)
- Quatro confundidores investigados: resolução, formato, perfil ICC e
  **densidade de linhas**
- Multi-semente de T01 (Etapa 4) — AUC 0,9964 ± 0,0002
- **RNF01 completo** — soma 17,871 s contra o limite de 30 s
- **T02 verificada contra o valor publicado** (Etapa 3)
- **Os três extratores de T04**, e as três representações medidas
- **T05 medida com as quatro fontes completas**

---

## Decisões tomadas

### A campanha em escala não será executada

Decidido em 16/08/2026. Levaria de 2,5 a 4 dias e repetiria as mesmas medições
com mais dados. O corpus e os manifestos existem — a escala foi **viabilizada**,
que era o objetivo da Etapa 1 —, e a Seção 3 de `DECISOES_METODOLOGICAS.md` já
mostra que o colapso OOD persiste com 60.000 imagens de treino, isto é, não é
efeito de escala insuficiente.

**Reversível:** `scripts/pipeline_campanha.py --corpus data/corvi2024_escala
--prefixo escala`.

### T04 entrou na interface — decisão de 17/08 revertida em 30/08

Decidido em 17/08/2026 manter T04 fora da tela, por estimativa de 1,5 GB de
modelos e 60 a 90 s de carga por imagem, acima do teto de 30 s do RNF01.

**A estimativa nunca fora medida, e estava errada:** 883 MiB e ~10 s. Com um
serviço residente no WSL2, T04 passou a pontuar imagens arbitrárias.

| | Estimado | Medido |
|---|---|---|
| VRAM | ~1,5 GB | 883 MiB |
| Carga | 60–90 s | ~10 s |
| RNF01 com T04 | violaria | **20,03 s** (teto 30 s) |

Construído:

| Arquivo | O quê |
|---|---|
| `scripts/wsl/servico_t04.py` | serviço residente, dois processos |
| `scripts/wsl/servico_t04_controle.sh` | iniciar/parar/consultar no WSL |
| `scripts/verificar_paridade_servico_t04.py` | valida contra os escores da dissertação |
| `scripts/medir_rnf01_com_t04.py` | remede o RNF01 pelo caminho da interface |

**Como subir** (as duas portas, com os dispositivos casados ao lote):

```powershell
# 8404: objeto-sombra em GPU
Start-Process wsl.exe -ArgumentList "-d","Ubuntu-24.04","-u","root","--",
  "/root/geo/bin/python","-u","/mnt/c/.../scripts/wsl/servico_t04.py",
  "--porta","8404","--dispositivo","cuda","--representacoes","object_shadow"

# 8405: campos e retas em CPU, com a placa escondida
Start-Process wsl.exe -ArgumentList "-d","Ubuntu-24.04","-u","root","--","env",
  "CUDA_VISIBLE_DEVICES=","/root/geo/bin/python","-u","/mnt/c/.../servico_t04.py",
  "--porta","8405","--dispositivo","cpu",
  "--representacoes","perspective_fields,line_segment"
```

Depois, a interface com `TCC3_T04_SERVICO=1`. **Sem essa variável nada muda** —
T04 volta a se declarar indisponível, como antes.

Paridade: campos e retas exatos até 1e-16; objeto-sombra em 2,3e-04, resíduo de
cuDNN com lote 1 contra lote 128. Justificativa completa na seção 5 de
`DECISOES_METODOLOGICAS.md`.

**O resultado não muda:** T04 mede 0,53 e a caixa exibe ruído. O ganho é de
demonstrabilidade da arquitetura de quatro fontes.

---

## Armadilhas conhecidas

Registradas porque cada uma já custou tempo.

1. **Compile antes de rodada longa.** Um `SyntaxError` custou 12 h de janela.
   `python -m compileall scripts src app tests`.
2. **Processos longos morrem sem traceback**, com log de erro vazio. Ocorreu
   cinco vezes. O Agendador protege contra o fechamento do shell, mas as tarefas
   estão em "Interativo apenas"; em 15/08 duas pararam às 23h55 sem causa
   determinada. Verificar o que está vivo antes de supor progresso.
3. **`schtasks /end` não derruba os filhos.** A GPU seguiu com 5,5 GB ocupados
   depois de encerrar a tarefa; só `taskkill /PID <pid> /T /F` liberou.
4. **Não sobrescrever os pesos do protocolo padrão.** Variantes devem usar
   `TCC3_WEIGHTS_DIR`, como fazem `pipeline_pos_multiseed.py` e
   `pipeline_t04_fusao.py`.
5. **O cache de escores não sabe que os dados de entrada mudaram.** Ao trocar os
   escores de T04 de um componente para os três, a campanha reaproveitou o
   `.npy` antigo — a forma batia — e reportou o valor errado em silêncio. Ao
   mudar a origem de um escore, apagar
   `results/scores/<protocolo>__<tecnica>__*`.
6. **Comparar modelos ajustados em ocasiões distintas mede o reajuste**, não a
   mudança que se quer testar. Um confronto assim sugeriu ganho significativo de
   T04 na fusão (p < 0,0001) que sumiu ao ajustar os dois na mesma validação
   (p = 0,182).
7. **O cache de escores colide entre manifestos.** A chave é
   `{protocolo}__{técnica}__{condição}__{split}`, e o protocolo é `ood` tanto
   para `manifesto30k_ood` quanto para `manifesto30k_ood_familias`. Hoje não
   causa erro porque as partições têm tamanhos diferentes e a guarda de forma em
   `runner.py:119` rejeita o incompatível — mas uma execução sobrescreve o cache
   da outra.
8. **T02 exige GPU livre.** As extrações de T04 no WSL2 também ocupam a placa e
   não convivem na RTX 3060 de 6 GB. Retas e campos de perspectiva rodam em CPU
   com `CUDA_VISIBLE_DEVICES` vazio, e aí sim convivem com treino na GPU.
9. **`wmic` não existe** nesta build do Windows 11.
10. **A interface não é afetada** pelos treinos: `run_multi_seed` grava em
    `weights/t01_seed<N>.pt`, sem tocar em `t01_cooccurrence.pt`.
11. **Um processo não serve as três representações de T04.** O PointNet dos
    autores (`lines_model.py:39-41`) manda a matriz identidade para CUDA sempre
    que `torch.cuda.is_available()` for verdadeiro — não quando o modelo está na
    GPU. O contorno é `CUDA_VISIBLE_DEVICES=""`, que vale para o processo
    inteiro e conflita com o objeto-sombra, que roda em GPU. Daí dois serviços.
12. **`data/amostra` não é dado real.** É o corpus de verificação funcional de
    `make_sample_corpus.py`: ruído 1/f em `real/`, artefato periódico plantado
    em `fake/`. As pastas levam os nomes de `REAL_SOURCES` e `ALL_GENERATORS`,
    e **`fodb`, `imagenet`, `open_images` e `raise` não existem em nenhum outro
    lugar do disco** — quem varrer por nome de fonte cai nelas. Custou duas
    medições e uma conclusão errada em 31/08. Hoje há
    `AVISO_CORPUS_SINTETICO.txt` na raiz do diretório, e
    `medir_fpr_t01_fontes.py` recusa qualquer caminho sob esse marcador.

13. **Medição sem controle não acusa o próprio erro.** A varredura das 19
    pastas não continha nenhuma entrada de valor conhecido, e por isso não teve
    como detectar que quatro vinham de outro corpus. A medição seguinte trazia a
    COCO como controle e detectou na primeira execução. **Toda medição nova
    carrega ao menos uma entrada cujo valor já está publicado no capítulo.**

14. **`--refit-fusion` grava por cima de um modelo reportado.** Ele salva em
    `WEIGHTS_DIR/t05_fusion.pkl`, que é **a variante B que a interface carrega
    quando a T04 está fora do ar**. O comando registrado na seção 2b, rodado
    como estava escrito, destruiria esse modelo. Qualquer reajuste precisa de
    `TCC3_WEIGHTS_DIR` apontando para diretório próprio — é o que o
    `recalibrar_t05_fusion.py` faz. Descoberto em 31/08 antes de morder.

15. **Automação de teclado não inicia depuração no VS Code.** `SendKeys` para a
    janela não dispara o `F5`: a tecla cai no editor e nenhuma sessão começa.
    Tentado em 02/09. Para subir a interface sem interação, use o caminho de
    terminal do `docs/COMO_RODAR.md`.

16. **Processo lançado de dentro do `wsl.exe` não sobrevive ao retorno da
    chamada** — nem com `setsid`, nem com `nohup`. Morre sem escrever no log,
    parecendo a armadilha nº 2. O que segura é lançar pelo lado Windows com
    `Start-Process wsl.exe ...`, que mantém um processo vivo no Windows
    ancorando o do WSL. Vale igual para o Gradio: `python app/gradio_app.py`
    em segundo plano da sessão morre junto com ela.
