# Estrutura do repositório

Reorganizado em 03/09/2026. Este documento é o mapa e, principalmente, o
registro do que **não** foi renomeado e por quê.

---

## O mapa

```
tcc3-deteccao-imagens-ia/
├── codigo/                       o pacote importável (era src/)
│   ├── configuracao.py           constantes, hiperparâmetros e CAMINHOS
│   ├── sementes.py               determinismo (RNF02)
│   ├── metricas.py               AUC, acurácia, F1, FPR/FNR, DeLong, bootstrap
│   ├── graficos.py               ROC, AUC por gerador, heatmap de robustez
│   │
│   ├── preprocessamento/         ★ tudo que acontece ANTES das técnicas
│   │   ├── validacao_envio.py       1. o arquivo é aceitável? (RN01, RN02)
│   │   ├── normalizacao.py          2. 256 px, LANCZOS, recorte central, PNG
│   │   ├── envio_interface.py       3. a política de resolução por técnica
│   │   └── perturbacoes.py          protocolo de robustez (JPEG, ruído, escala)
│   │
│   ├── tecnicas/                 ★ uma pasta por técnica
│   │   ├── base.py               interface fit / predict_proba / score
│   │   ├── t01_coocorrencia/     __init__.py = a ficha da técnica
│   │   ├── t02_spai/             tecnica.py  = a implementação
│   │   ├── t03_benford/
│   │   ├── t04_geometria/
│   │   └── t05_fusao/
│   │
│   ├── corpus/                   montagem e particionamento (era src/data/)
│   │   └── manifesto.py          qual imagem vai para qual split
│   └── experimentos/
│       └── runner.py             orquestração dos três protocolos
│
├── interface/                    a aplicação Gradio (era app/)
├── automacao/                    scripts de preparação e execução (era scripts/)
│   └── wsl/                      extratores e serviços de T04 no WSL2
├── testes/                       148 testes (era tests/)
├── documentacao/                 este e os demais registros (era docs/)
│
├── data/                         o corpus — NOME MANTIDO, ver abaixo
├── pesos/                        modelos treinados (era weights/)
├── pesos_*/                      variantes isoladas (eram weights_*/)
├── resultados/                   saídas medidas (era results/)
└── externo/                      repositórios oficiais de T02 e T04 (era external/)
```

---

## As duas pastas que motivaram o rearranjo

### `codigo/preprocessamento/`

Existe porque o pré-processamento **não é detalhe de implementação**: é o que
decide se o que aparece na tela corresponde ao que foi medido.

Até 31/08/2026 a interface entregava o upload **cru** às cinco técnicas,
enquanto todos os números do Capítulo 4 vinham de imagens normalizadas. T01 e
T02 trocavam de lado conforme a resolução do envio — a mesma imagem sintética
dava 0,0% numa técnica e 99,9% na outra, e o inverso ao normalizar. O código
que conserta isso estava espalhado entre `scripts/normalize_corpus.py` e
`app/gradio_app.py`; agora está reunido, na ordem em que a imagem o atravessa.

A função `resize_and_center_crop` é a **mesma** para o corpus e para cada
upload. Essa igualdade é o que sustenta a correspondência entre a tela e o
capítulo, e é por isso que ela vive num só lugar: duplicá-la abriria espaço
para as duas divergirem em silêncio.

### `codigo/tecnicas/`

Cada técnica ganhou pasta própria. O `__init__.py` de cada uma traz a ficha —
o que a técnica mede, **em que resolução foi medida**, o resultado e a
fragilidade conhecida — e a implementação fica em `tecnica.py`.

Os arquivos não foram fatiados. O maior tem 493 linhas, abaixo do teto de 800,
e quebrá-los seria churn sem ganho. O que faltava não era tamanho menor: era
poder abrir a pasta de uma técnica e achar ali o que se sabe sobre ela.

---

## O que NÃO foi renomeado, e por quê

### `data/` continua em inglês — decisão deliberada

Os manifestos gravam **caminho absoluto** de cada imagem:

```
C:\Users\ferre\projects\tcc3-deteccao-imagens-ia\data\corvi2024_30k_norm\real\coco\coco_000000.png
```

São **132.640 linhas** nos dois manifestos de 30k, mais os três da escala.
Renomear a pasta obrigaria a reescrever o registro de qual imagem entrou em
qual split — isto é, a base de reprodução de todo número do Capítulo 4.

E `data/` **não é versionada**. Se a reescrita saísse errada, não haveria
`git show HEAD:` para socorrer — que foi exatamente o que salvou os números da
dissertação quando a armadilha nº 4 os sobrescreveu em 31/08.

Uma inconsistência de nome custa menos que um registro de reprodução mexido.
A decisão está registrada em `codigo/configuracao.py`, no comentário sobre
`DATA_DIR`, para quem estranhar o nome ao ler o código.

### Os nomes de arquivo de peso e de escore

`t05_fusion.pkl`, `t01_cooccurrence.pt`, `t04_escores_combined.csv` e afins
mantêm os nomes. São **arquivos existentes em disco**, alguns versionados, e
vários citados nominalmente em `RESULTADOS.md`. Renomeá-los quebraria a ligação
entre o que o texto afirma e o que existe.

### O conteúdo dos JSON de `resultados/`

Alguns gravam o caminho dos pesos usado **no momento da medição**:

```json
"pesos": "C:\...\tcc3-deteccao-imagens-ia\weights\projective_geometry\..."
```

Esse `weights\` está obsoleto e **foi deixado como está de propósito**. É o que
o caminho era quando a medição rodou. Reescrevê-lo faria o registro afirmar
algo que não aconteceu.

---

## O que o rearranjo consertou sem querer

**`src/data/` nunca esteve no git.** A regra `data/` do `.gitignore`, escrita
para o corpus, casava com qualquer diretório chamado `data` em qualquer
profundidade — e engolia o pacote de código em silêncio. Eram 590 linhas fora
do versionamento, entre elas `manifest.py` (442 linhas), justamente o código que
constrói os manifestos.

Duas coisas mudaram: o pacote virou `codigo/corpus/`, fora do alcance da regra,
e a regra foi ancorada na raiz (`/data/`), para não voltar a pegar subpastas.

---

## Como isto foi verificado

| Verificação | Resultado |
|---|---|
| Linha de base, antes de mover | 148 testes passam em 200 s |
| Suíte após o rearranjo | **148 passam** |
| `compileall` (armadilha nº 1) | limpo |
| 10 tarefas do Agendador | recriadas; **as 10** apontam para arquivo existente |
| Serviços T04, parados e ressubidos | no ar em 15 s, dispositivos corretos |
| **Paridade de T04 contra os escores publicados** | **36 comparações, 0 divergentes** |

A paridade é o controle que importa: `automacao/verificar_paridade_servico_t04.py`
compara o serviço contra `resultados/t04_escores_componentes.csv`, que é a fonte
dos valores publicados (AUC 0,5384 / 0,5355 / 0,5187). Diferença máxima de
8,14e-05 em objeto-sombra — dentro do resíduo de cuDNN já documentado — e da
ordem de 1e-17 nas outras duas. **Nenhum valor medido mudou.**

O backup dos XML das tarefas do Agendador, como estavam antes, ficou em
`.vscode/backup_tarefas_agendador/`.

---

## A armadilha que o rearranjo revelou

Renomear pastas quebra caminhos de três formas, e só a primeira é óbvia:

1. **Prefixo de texto** — `"scripts/foo.py"` numa docstring ou num `.bat`.
2. **Componente de `Path`** — `RAIZ / "weights" / "projective_geometry"`.
   Não tem barra, então nenhuma busca por `weights/` o encontra. Foi isto que
   derrubou os serviços T04 na primeira tentativa de ressubir: o serviço
   carregava pesos de um diretório que tinha deixado de existir, e a falha
   apareceu só como "não responderam em 90 s".
3. **Fora do repositório** — as tarefas do Agendador guardam caminho absoluto
   no próprio Windows, e nenhuma busca no código as alcança.

Vale como regra para a próxima vez: **depois de renomear, tentar subir o
sistema inteiro**. A suíte de testes passou nos 148 casos com os serviços T04
apontando para um caminho inexistente — porque os testes não sobem os serviços.
