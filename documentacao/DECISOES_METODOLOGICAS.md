# Decisões metodológicas não previstas no TCC 2

Registro das alterações metodológicas introduzidas durante a execução (TCC 3)
que **não** constavam do projeto aprovado no TCC 2. Cada uma traz o motivo, a
evidência empírica que a exigiu e o impacto sobre a interpretação dos resultados.

Este documento existe para que nenhuma dessas mudanças chegue à banca como
detalhe de implementação silencioso. Todas devem ser declaradas no Capítulo 4.

---

## 1. Calibração de T05 sobre gerador *held-out*

### O que o TCC 2 previa

A Seção 3.5.2 descreve T05 como ajustada sobre os escores das demais técnicas,
sem especificar a partição. A implementação inicial usou o subconjunto de
**validação**, escolha natural por não participar do ajuste de parâmetros de
T01 e T03.

### Por que isso não funcionou

O protocolo OOD expôs o problema. Com T05 calibrada sobre a validação
*in-distribution*:

| Protocolo | T01 | T02 | T03 | **T05** |
|---|---|---|---|---|
| Padrão (in-distribution) | 1,0000 | 0,9993 | 0,9850 | **1,0000** |
| OOD (13 geradores não vistos) | 0,6369 | — | 0,5830 | **0,5880** |

**T05 (0,5880) ficou abaixo de T01 isolada (0,6369).** A fusão, que deveria
elevar a capacidade discriminativa, piorou o resultado fora da distribuição.

A causa é a distribuição dos escores de calibração. No subconjunto de validação
as três técnicas-base acertam quase tudo — AUC entre 0,985 e 1,000. O
classificador de fusão aprende, portanto, pesos apropriados a um regime em que
**todas as fontes são confiáveis e concordam entre si**. Esse regime não se
repete no OOD, onde cada técnica degrada de forma diferente: T01 mantém alguma
capacidade (0,6369), T03 fica quase no acaso (0,5830), e as discordâncias
passam a carregar informação que a calibração in-distribution nunca observou.

Em outras palavras: a fusão foi calibrada num cenário fácil e aplicada num
cenário difícil, sem nunca ter visto como suas fontes se comportam quando erram.

### O que foi implementado

Um **terceiro conjunto disjunto**, reservado exclusivamente à calibração de T05:

```
treino          latent_diffusion            (tecnicas-base T01 e T03)
calibracao      glide + 30% das reais       (apenas T05)
avaliacao       12 geradores restantes      (metricas reportadas)
```

- `codigo/configuracao.py` — `FUSION_CALIBRATION_GENERATORS` e `FUSION_CALIBRATION_REAL_RATIO`
- `codigo/corpus/manifesto.py` — `reserve_fusion_calibration()`, aplicada após `ood_split()`
- `codigo/experimentos/runner.py` — `fit_fusion(split=...)`, generalizada para qualquer partição
- `automacao/prepare_dataset.py` — reserva automática no protocolo OOD; `--no-fusion-split` desativa
- `automacao/run_experiments.py` — `--fusion-split {val,fusion}`

O gerador de calibração (`glide`) não participa do treinamento das técnicas-base
nem entra nas métricas finais. As imagens reais de calibração são sorteadas com
semente fixa e removidas do conjunto de avaliação, de modo que os três conjuntos
sejam mutuamente disjuntos.

### Como reportar no Capítulo 4

As duas variantes devem ser apresentadas lado a lado, não apenas a melhor:

```bash
# Variante A - calibracao in-distribution (a do projeto original)
python automacao/run_experiments.py --protocol ood --refit-fusion --fusion-split val

# Variante B - calibracao sobre gerador held-out
python automacao/run_experiments.py --protocol ood --refit-fusion --fusion-split fusion
```

A comparação entre A e B **é um resultado do trabalho**, não um detalhe de
ajuste: mostra que o ganho de uma arquitetura de fusão tardia depende
criticamente de a calibração refletir o regime de aplicação. Reportar apenas a
variante B, omitindo que a A foi tentada primeiro e falhou, esconderia a
descoberta mais informativa.

### Limitação que permanece

A avaliação passa a cobrir 12 geradores em vez de 13, já que `glide` foi
reservado. O resultado deixa de ser diretamente comparável, em número de
geradores, ao benchmark de Karageorgiou et al. (2025). A alternativa seria
validação cruzada *leave-one-generator-out*, com 13 calibrações independentes —
mais robusta, porém com custo de inferência 13 vezes maior, inviável no prazo
e no hardware disponíveis. Fica registrada como trabalho futuro.

---

## 2. Protocolo OOD mais rigoroso que o previsto

### O que o TCC 2 previa

A Seção 3.6.1 definia treinamento sobre os geradores de difusão e avaliação
sobre GigaGAN, Midjourney v5 e v6.1 — três geradores retidos.

### O que foi implementado

O corpus de treinamento de Corvi et al. (2024) provém de **um único modelo de
difusão latente**, distinto dos 13 geradores do benchmark de avaliação. Isso
permitiu um protocolo mais rigoroso: treinar em um gerador e avaliar nos treze,
**nenhum** visto no treinamento.

`codigo/corpus/manifesto.py::ood_split()` implementa essa divisão, com
`TRAINING_GENERATORS` e `BENCHMARK_GENERATORS` em `codigo/configuracao.py`.

### Impacto

A queda de AUC observada (36 a 41 pontos percentuais) é maior do que seria sob
o protocolo original, porque a separação entre treino e teste é mais completa.
O indicador do risco R01 previa queda superior a 10 p.p.; o valor medido é três
a quatro vezes esse limiar. **Comparações com a literatura devem considerar que
este protocolo é mais severo que o usual.**

---

## 3. Escala reduzida do corpus real

### O que o TCC 2 previa

180.000 imagens reais e 180.000 sintéticas (Seção 3.2).

### O que foi possível

O arquivo `latent_diffusion_trainingset.zip` distribui as 180.000 sintéticas,
mas apenas as **listas** das imagens reais (`real_coco.txt`, `real_lsun.txt`),
que apontam para COCO train2017 e LSUN — obtidos separadamente. ImageNet exige
conta Kaggle, RAISE-1k exige formulário de cadastro, e LSUN não foi baixado.

O corpus efetivo usa **5.000 imagens reais do COCO val2017** como única fonte
real, com 5.000 sintéticas de difusão latente para manter proporção 1:1.
Registrado em `data/corvi2024/corpus_info.json`.

### Onde está o gargalo

O arquivo distribui as sintéticas, mas para as reais entrega apenas listas de
nomes. As contagens exatas:

| Conteúdo | Quantidade | Situação |
|---|---|---|
| Sintéticas `train/` | 180.000 | incluídas no arquivo |
| Sintéticas `valid/` | 20.000 | incluídas no arquivo |
| `real_coco.txt` | 90.000 | **apenas a lista** — aponta para COCO train2017 |
| `real_lsun.txt` | 90.000 | **apenas a lista** — aponta para LSUN |

A Seção 3.5.1 exige proporção 1:1 entre as classes. Com apenas 5.000 imagens
reais disponíveis (COCO val2017), usar as 180.000 sintéticas produziria
desbalanceamento de 36:1 — disparando o risco R07 e tornando a acurácia
ininterpretável. O número de sintéticas foi portanto limitado ao número de
reais disponíveis, e não por escolha metodológica.

### Escalas avaliadas

| Escala | Reais | Sintéticas | Custo estimado |
|---|---|---|---|
| Inicial | 5.000 | 5.000 | ~3 h |
| **Adotada** | **30.000** | **30.000** | **~8 h** |
| Cheia (COCO) | 90.000 | 90.000 | 2 a 3 dias |
| Protocolo integral | 180.000 | 180.000 | exige LSUN, não baixado |

A escala adotada exigiu o download de COCO train2017 (18 GB). A escala cheia foi
descartada pelo custo: o treinamento de T01 sobre 63.000 imagens e a busca em
grade de T03 sobre 90.000 × 540 características ultrapassariam o prazo
disponível. O protocolo integral exigiria adicionalmente LSUN.

### Impacto — e é sério

**Viés de fonte única.** Com apenas COCO como fonte real, um detector pode
aprender a distinguir "COCO versus difusão latente" em vez de evidência forense
genuína. É exatamente o viés de conjunto de dados discutido na introdução do
TCC 2. Obter ImageNet, RAISE, FODB e Open Images é o próximo passo para
eliminar a limitação.

**Tamanho da amostra de treino.** Os primeiros resultados foram obtidos com
5.000 imagens sintéticas de treino — 36 vezes menos que as 180.000 disponíveis.
A queda de AUC de 36 a 41 p.p. medida no protocolo OOD **não pode ser atribuída
com segurança à incapacidade de generalizar** enquanto não for reproduzida em
escala maior: com poucos exemplos, a rede tem mais incentivo a memorizar o
artefato de um único gerador. Esse é o motivo do aumento para 30.000.

---

## 4. Normalização de resolução e formato — confundidor detectado

### Como o problema apareceu

A verificação de integridade da Etapa 1 sinalizou **16.638 de 22.638 imagens
(73%) com resolução diferente de 256 × 256**. O aviso constava do relatório
(`data/integridade_ood.json`, campo `wrong_size`) desde a primeira execução,
mas foi inicialmente atribuído ao desbalanceamento entre geradores e não
investigado. Só foi diagnosticado quando T02 passou a falhar de forma
reprodutível no protocolo OOD.

Composição real do corpus antes da normalização:

| Grupo | Resolução | Formato |
|---|---|---|
| `real/coco` | ~640 × 427 (variável) | JPEG |
| `fake/latent_diffusion` (treino) | 256 × 256 | PNG |
| `fake/glide` | 256 × 256 | PNG |
| `fake/flux` | 1024 × 768 | PNG |
| `fake/midjourney_v6_1` | 1024 × 1024 e maiores | PNG |
| `fake/adobe_firefly` | **2304 × 1792** | PNG |

### Por que invalida os resultados anteriores

No conjunto de treinamento, **todas** as sintéticas têm 256 × 256 em PNG e
**todas** as reais têm ~640 × 427 em JPEG. São dois confundidores sobrepostos —
resolução e formato de compressão — e qualquer um deles, isoladamente, permite
separar as classes sem aprender evidência forense.

O valor `AUC = 1,0000` obtido por T01 no protocolo in-distribution deve ser
lido sob essa ressalva: não há como distinguir, naquela medição, capacidade
forense de exploração do confundidor.

É precisamente o viés de conjunto de dados discutido na introdução do TCC 2 —
detectores aprendendo correlações espúrias associadas a formato e compressão em
vez de características forenses genuínas. O experimento reproduziu o problema
em vez de medi-lo.

### Efeito prático sobre T02

As imagens do Adobe Firefly têm 4,1 megapixels, 63 vezes mais que 256 × 256. O
SPAI processa em resolução nativa por construção (*Spectral Context Attention*),
de modo que o custo de inferência e o consumo de memória escalam com a área. A
falha ocorria de forma consistente por volta da imagem 700–800 do conjunto de
teste OOD — exatamente a fronteira entre as 750 imagens reais e o início do
`adobe_firefly` na ordenação do manifesto.

### O que foi implementado

`automacao/normalize_corpus.py`:

1. **Redimensionamento com preservação de proporção** — o menor lado vai a 256 e
   o excedente é removido por recorte centralizado. Um resize direto para
   256 × 256 distorceria a geometria de imagens não quadradas, alterando as
   relações de perspectiva e as estatísticas locais que as técnicas analisam.
   A reamostragem usa LANCZOS, que preserva melhor o conteúdo de alta
   frequência onde residem os artefatos explorados por T01 e T03.
2. **Formato único PNG** — elimina a segunda compressão JPEG que distinguiria
   as imagens reais das sintéticas.

### Consequências

- **Os três protocolos precisam ser reexecutados.** Todos os resultados
  anteriores à normalização ficam invalidados, incluindo a queda de AUC de
  36 a 41 p.p. no OOD: parte dessa queda pode decorrer da mudança de resolução
  entre treino e teste, e não de incapacidade de generalizar.
- **Limitação declarada:** avaliar o SPAI apenas em 256 × 256 subestima o
  método, cuja capacidade de operar em resolução arbitrária é uma de suas
  contribuições. A restrição decorre dos 6 GB de VRAM disponíveis e deve
  constar da análise dos resultados.
- **Limitação residual:** as imagens reais do COCO já carregam artefatos JPEG
  no conteúdo, anteriores à normalização. Converter para PNG evita uma segunda
  compressão, mas não remove a primeira. Uma alternativa seria aplicar JPEG
  uniforme a ambas as classes, aproximando o cenário de redes sociais — o que o
  protocolo de robustez já explora separadamente.

---

## 4b. Terceiro confundidor: perfis ICC embutidos

Detectado durante a montagem do corpus em escala, ao investigar por que uma
imagem havia ficado ilegível.

### Como apareceu

A verificação de integridade acusou dois arquivos corrompidos em 192.638. Um
tinha 0 byte — escrita interrompida, tratada com gravação atômica. O outro,
`coco_082449.png`, tinha 1.459.610 bytes contra os ~100 KB típicos e era
reproduzível, portanto não era escrita parcial. A inspeção dos chunks revelou:

```
IHDR      13 bytes
iCCP   1.367.158 bytes      <- perfil de cor ICC
IDAT      92.371 bytes      <- a imagem em si
```

O JPEG de origem no COCO trazia um perfil ICC de 1,3 MB, que o Pillow copiou
para o PNG. Ao reabrir, a proteção contra bomba de descompressão o rejeita com
`Decompressed Data Too Large`.

### A medição que importa

| Grupo | Imagens com perfil ICC |
|---|---|
| **real/coco** | **51,0%** |
| latent_diffusion | 0,0% |
| glide | 0,0% |
| dalle3 | 0,0% |
| midjourney_v5 | 0,0% |

Metade das imagens reais carrega um marcador que **nenhuma** sintética possui.
Estruturalmente é o mesmo problema da resolução e do formato: um atributo que
separa as classes sem relação alguma com síntese.

### Por que os resultados **não** ficam invalidados

Diferente do confundidor de resolução, este **não chega ao modelo**. Verificado
empiricamente: carregou-se cada imagem com perfil, regravou-se sem ele e
compararam-se os arrays de pixels.

```
5/5 imagens: pixels idênticos, diferença máxima 0
```

O Pillow não aplica o perfil ICC na decodificação, de modo que T01, T02 e T03
recebem exatamente os mesmos dados com ou sem o chunk. **A rodada de 30k
permanece válida** — o corpus dela apresenta a mesma proporção (45,3% contra
0%), sem efeito sobre as métricas.

### O que foi feito

Todo metadado passou a ser descartado na gravação
(`normalizada.info.pop("icc_profile", None)`), e as 90.000 imagens reais do
corpus em escala foram regeradas. Motivo da correção, mesmo sem contaminar
resultados:

1. Um perfil de 1,3 MB tornou uma imagem irrecuperável.
2. Higiene de corpus: outro carregador, ou uma análise futura, poderia explorar
   o marcador sem que se percebesse.
3. Espaço: perfis de mais de 1 MB em imagens de 100 KB.

### Registro para o Capítulo 4

É o **terceiro** confundidor encontrado neste corpus, depois de resolução e
formato. Os dois primeiros invalidaram uma rodada inteira; este não. A diferença
— um altera os pixels, o outro não — só pôde ser estabelecida por medição, não
por inspeção do código, e é o tipo de verificação que a montagem de qualquer
corpus forense deveria incluir.

---

## 5. T04 não integrada — e replicação parcial do componente objeto-sombra

Registro completo em [`externo/CONTRATO.md`](../external/CONTRATO.md).

### O bloqueio, com evidência no código oficial

Os três classificadores de T04 **não recebem pixels**. Cada um carrega uma
representação geométrica já extraída por um modelo auxiliar que o repositório
não distribui:

| Componente | Evidência no código oficial | Entrada real | Extrator exigido |
|---|---|---|---|
| perspective_fields | `fields_dataset.py:31` — `torch.load(field_path)` | tensores `.pt` com `pred_latitude_original` e `pred_gravity_original` | PerspectiveFields (Jin et al.) → detectron2 |
| object_shadow | `dataset.py:20-21` — `Image.open(shadow/object)` | máscaras de sombra e de objeto | SSISv2 → detectron2/AdelaiDet |
| line_segment | `lines_dataset.py:22` — `image_path_to_lines[...]` | segmentos pré-extraídos | detector de retas |

O próprio artigo declara: *"All three classifiers are denied access to image
pixels, and look only at derived geometric features."*

Escrever o adaptador `infer.py` que `codigo/tecnicas/t04_geometria/tecnica.py` procura
**não resolveria**: não haveria o que alimentar ao classificador. A mensagem de
indisponibilidade foi corrigida para dizer isso — antes ela reportava "script de
inferência ausente", o que sugeria, incorretamente, um arquivo faltando.

### O que torna uma replicação parcial possível

O `object_shadow/README.md` aponta para um segundo repositório,
[`Projective-Geometry-OS`](https://huggingface.co/datasets/amitabh3/Projective-Geometry-OS),
com as **máscaras objeto-sombra já extraídas** para o conjunto de teste dos
autores (22 GB, um arquivo por gerador). Isso permite avaliar o classificador
oficial de objeto-sombra sem executar o SSISv2.

Implementado em `automacao/replicar_t04_object_shadow.py`.

### Limites explícitos dessa replicação

- O corpus é o **dos autores** (Kandinsky), não o de Corvi et al. (2024). A
  métrica **não** entra na tabela comparativa com T01, T02, T03 e T05.
- Não participa da fusão T05 nem da interface: para imagens arbitrárias, a
  extração das máscaras continua exigindo SSISv2.
- Cobre **um** dos três componentes. Perspective fields e line segments
  permanecem sem representações publicadas — seus datasets estão em Google
  Drive e não foram verificados.

O valor é de verificação: confirma que os pesos oficiais reproduzem o
desempenho publicado, sustentando a descrição de T04 no Capítulo 2 com execução
própria em vez de apenas citação.

### Por que um adaptador, e não o `test.py` oficial

O script oficial divide caminhos com `split("/")` em quatro pontos
(`test.py` 19, 23, 81 e `dataset.py` 28). No Windows o `glob` devolve barras
invertidas, de modo que `dataset.py:28` produziria a string inteira em vez do
nome da classe e a rotulagem falharia. O adaptador reimplementa a leitura
mantendo arquitetura, pesos e composição dos subconjuntos originais, e preserva
o repositório oficial intacto — a replicação segue auditável.

A técnica reporta indisponibilidade na interface e as demais seguem operando
(RN07).

### Atualização: o bloqueio de objeto-sombra foi levantado

O que estava bloqueado era o **extrator**, não o classificador. O SSISv2 foi
posto para rodar em GPU no WSL2 — ambiente que a Seção 3.2 do TCC 2 já admite —
e o registro completo, com os oito obstáculos vencidos, está em
[`T04_AMBIENTE_WSL2.md`](T04_AMBIENTE_WSL2.md).

Com isso, a limitação "não entra na tabela comparativa nem na fusão T05" deixa
de valer para este componente. As máscaras do corpus deste trabalho foram
extraídas e classificadas, e T04 passa a ocupar sua coluna em
`codigo/tecnicas/t05_fusao/tecnica.py`, antes sempre NaN.

**Três ressalvas que precisam acompanhar o número onde ele aparecer:**

1. **É um terço da técnica.** Campos de perspectiva e segmentos de reta seguem
   sem extrator. `is_available()` passou a devolver disponibilidade *parcial*, e
   a agregação por `nanmean` ignora as duas ausentes — o mesmo mecanismo de RN07.
   Reporta-se como **T04 (objeto-sombra)**, nunca como T04.
2. **É transferência entre domínios.** Os pesos são os oficiais, treinados em
   Kandinsky e aplicados sem reajuste, como a Etapa 2 exige. Desempenho abaixo
   do publicado não indica falha de replicação.
3. **O resultado é negativo.** Sobre pares com conteúdo e com os pesos
   `outdoor` — o recorte comparável ao da seção 4.7 — a AUC é 0,5797, contra
   0,8216 do mesmo classificador sobre o corpus de origem. Em `combined` e
   `indoor` fica em acaso. As verificações que descartam erro de mapeamento de
   classes, falha do extrator e o confundidor das máscaras vazias estão na
   seção 4.8 de [`RESULTADOS.md`](RESULTADOS.md), junto com o confundidor que
   permanece aberto.

   Na fusão, T05 se manteve em 0,9996 e atribuiu a T04 peso −0,0403: aprendeu a
   ignorá-la. Ver seção 4.8.1.

### T04 permanece fora da interface — e por qual motivo

Decidido em 16/08/2026, depois de os três extratores estarem funcionando.

Com SSISv2, PerspectiveFields e DeepLSD operando no WSL2, T04 **passou a ser
tecnicamente executável** sobre uma imagem arbitrária enviada pela tela. Isso
muda a natureza da limitação do RF02, e a mudança precisa ser dita: antes T04
era indisponível porque o método não podia ser executado; agora é indisponível
por **arquitetura de execução**.

O obstáculo apontado à época era o custo de carga. Os três modelos somam cerca
de 1,5 GB (SSISv2 587 MB, PerspectiveFields 798 MB, DeepLSD 100 MB) e vivem em
outro sistema operacional; estimou-se que uma chamada por imagem gastaria 60 a
90 s apenas carregando — acima do teto de 30 s do RNF01. Atender o requisito
exigiria um **serviço persistente no WSL2**.

**Optou-se por não construí-lo**, por três razões:

1. O comportamento atual **já cumpre** o previsto: RN07 determina que a falha de
   um módulo não interrompa os demais, e `is_available()` reporta
   indisponibilidade com motivo descritivo — coberto por teste automatizado.
2. T04 mede cerca de **0,53 de AUC** neste corpus (seções 4.8 e 4.12 de
   `RESULTADOS.md`). A caixa na tela exibiria ruído, o que é pior que uma
   indisponibilidade honesta: sugeriria ao usuário uma evidência que não existe.
3. O esforço se justificaria por completude formal do RF02, não por valor
   analítico — e há itens de maior retorno em aberto.

### Decisão revista em 30/08/2026: a estimativa estava errada

A justificativa acima foi construída sobre uma estimativa **nunca medida**. Ao
medir, o argumento central não se sustentou:

| | Estimado em 17/08 | Medido em 30/08 |
|---|---|---|
| VRAM dos três modelos | ~1,5 GB | **883 MiB** |
| Tempo de carga | 60 a 90 s | **~10 s** |

O erro no tempo é de quase uma ordem de grandeza. Com os modelos residentes, o
custo por imagem cai para a inferência, e o serviço foi construído:
`automacao/wsl/servico_t04.py`.

**São dois processos, não um.** A extração em lote não usou o mesmo dispositivo
para as três representações — o que só aparece nos logs, não no código:
objeto-sombra em GPU, campos e retas em CPU. E CPU não é opcional para as retas:
o PointNet dos autores (`lines_model.py:39-41`) move a matriz identidade para
CUDA sempre que `torch.cuda.is_available()` for verdadeiro, de modo que o único
contorno é `CUDA_VISIBLE_DEVICES=""` — que vale para o processo inteiro. Um
processo só não consegue usar a GPU para uma representação e escondê-la para
outra.

**Paridade verificada** (`automacao/verificar_paridade_servico_t04.py`, 120
comparações sobre 40 imagens): campos e retas exatos até 1e-16; objeto-sombra
com resíduo de 2,3e-04, caracterizado como efeito de a cuDNN escolher algoritmos
diferentes para lote 1 (serviço) e lote 128 (`avaliar_t04_corpus.py`).

**RNF01 remedido**, pelo caminho real da interface:

| | Sem T04 | Com T04 |
|---|---|---|
| T01 | 4,56 s | 4,56 s |
| T02 | 14,00 s | 13,89 s |
| T03 | 0,08 s | 0,08 s |
| T04 | — | **1,50 s** |
| **Total** | 18,84 s | **20,03 s** |

Margem de 9,97 s contra o teto de 30 s. O custo de T04 é de cerca de 1,9 s, e
não há disputa de GPU com T02 em regime.

Uma medição intermediária expôs um defeito que a decisão original não previa: a
**primeira** análise depois de a interface subir custava 40,5 s — acima do teto
—, por criação do contexto CUDA. Corrigido com uma inferência de aquecimento na
carga (`DetectionService._aquecer`), que traz a primeira imagem para 20,0 s.

**O que não muda:** a razão nº 2 acima continua valendo. T04 mede 0,53 neste
corpus, e a caixa na tela exibe ruído. O que se ganha é demonstrabilidade da
arquitetura de quatro fontes, com uma fonte fraca real em vez de simulada — e a
tela passa a corresponder ao que o Capítulo 4 descreve. A leitura do número
exibido deve vir acompanhada dessa ressalva.

O serviço é **opcional e desligado por padrão**: sem `TCC3_T04_SERVICO=1`, T04
volta a se declarar indisponível exatamente como antes.

O modo de escores pré-extraídos existe justamente para separar os dois usos: a
campanha experimental liga T04 por `TCC3_T04_ESCORES`, e a interface, que recebe
imagens arbitrárias, nunca o liga. Ver `codigo/tecnicas/t04_geometria/tecnica.py`.

Nada disso altera o status de T04 como técnica completa: ela segue não
replicada, e `externo/CONTRATO.md` continua valendo para os outros dois
componentes.

---

## 6. Ambiente de execução divergente

O Capítulo 3 indica Python 3.10, PyTorch 2.1 e CUDA 12.1. PyTorch 2.1 não
publica binários para Python 3.11+, tornando essa combinação não instalável.

Ambiente efetivo: **Python 3.11 + PyTorch 2.5.1 + CUDA 12.4**, com CUDA
confirmada na RTX 3060 de 6 GB.

Três pinos adicionais foram necessários, todos ocorrências do risco R02 e
documentados em `requirements.txt`:

| Pino | Motivo |
|---|---|
| `huggingface_hub==0.25.2` | Gradio 4.44 importa `HfFolder`, removido na versão 1.0 |
| `fastapi==0.112.4` | a partir de 0.113 o schema traz `additionalProperties` booleano, que `gradio_client` 1.3 não interpreta |
| `pydantic==2.9.2` | mesma causa |

A falha de FastAPI/pydantic se manifesta como `"When localhost is not
accessible, a shareable link must be created"` — mensagem que não indica a
causa real, pois o erro de schema ocorre no healthcheck de inicialização.

O SPAI roda em ambiente virtual próprio (`externo/spai/.venv`), invocado por
subprocesso. Seu `requirements.txt` oficial está incompleto: `filetype` não
consta e é importado por `spai/__main__.py`.

---

## 7. Estado dos pesos e rastreabilidade entre rodadas

O corpus foi reconstruído duas vezes, e os modelos treinados sobre cada versão
**não são intercambiáveis**. Registrar isso importa porque a interface carrega
o que estiver em `pesos/`, sem distinguir a procedência.

| Rodada | Corpus | Imagens | Situação |
|---|---|---|---|
| 1 | `data/corvi2024` | 22.638 | Invalidada pelo confundidor da seção 4 |
| 2 | `data/corvi2024_30k_norm` | 72.638 | Normalizada: 256 × 256, PNG, sem confundidor |

Os pesos da rodada 1 foram preservados em `pesos/backup_22k_confundido/`. Não
devem ser usados para produzir resultados; servem apenas como referência do
efeito do confundidor — a comparação entre as duas rodadas é, ela própria,
evidência do viés de conjunto de dados discutido na seção 4.

### Verificação de integridade da rodada 2

A indexação do corpus normalizado confirmou a eliminação do confundidor:

```
Total de imagens ......... 72638   (30.000 reais + 42.638 sintéticas)
Arquivos corrompidos ..... 0
Resolução inesperada ..... 0        <- era 16.638 de 22.638
Grupos de duplicatas ..... 0
```

O relatório ainda marca `REQUER ATENCAO` por dois motivos **esperados e
intencionais**, que não indicam defeito:

- **Desbalanceamento entre geradores (885%)** — `latent_diffusion` tem 30.000
  imagens por ser o gerador de *treinamento*, enquanto os 13 geradores do
  *benchmark* têm 1.000 cada, conforme a Seção 3.2. Os papéis são distintos e a
  comparação direta entre eles não se aplica.
- **`latent_diffusion` como gerador não previsto** — ele não integra a lista de
  geradores do benchmark justamente porque é o conjunto de treino.

### Falha de execução da primeira tentativa da rodada 2

A primeira execução de `automacao/pipeline_30k.py` (PID 30040) concluiu a
normalização e os dois manifestos, mas as quatro etapas de experimento
abortaram em segundos com `SyntaxError: unterminated string literal` em
`automacao/run_experiments.py:241`. A causa foi uma quebra de linha literal
inserida dentro de duas *f-strings* durante uma edição anterior do arquivo —
o mesmo defeito que já havia ocorrido em `codigo/tecnicas/t02_spai/tecnica.py`.

Consequência prática: nenhum experimento da rodada 2 chegou a executar na
primeira tentativa; a normalização e os manifestos, porém, foram aproveitados.

Medida adotada: `python -m compileall scripts src app tests` passou a ser
executado antes de disparar qualquer rodada longa. O defeito era detectável em
menos de um segundo, mas custou o intervalo inteiro entre o fim da normalização
e a verificação seguinte.
