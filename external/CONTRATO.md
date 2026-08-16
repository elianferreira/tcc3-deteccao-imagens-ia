# Contrato de integração com os pipelines oficiais (T02 e T04)

T02 e T04 são executadas em **modo de inferência**, por meio do código e dos
modelos oficiais dos respectivos autores. Como cada repositório exige um
ambiente Python próprio e mutuamente incompatível, a integração ocorre por
**subprocesso**, não por importação direta. Este documento fixa o contrato que
cada adaptador espera.

---

## T02 — SPAI

**Repositório:** <https://github.com/mever-team/spai> (Apache 2.0)
**Ambiente:** Python 3.11, PyTorch com CUDA 12.4
**Memória de GPU:** < 8 GB na inferência (compatível com a RTX 3060 de 6 GB)

O adaptador invoca a CLI oficial sem modificação:

```bash
python -m spai infer --input <input.csv> --output <output_dir>
```

- `--input` recebe um CSV com a coluna `image`, contendo um caminho absoluto por linha.
- `--output` recebe um diretório onde o pipeline grava um CSV de predições.

O adaptador (`src/techniques/t02_spai.py`) aceita as colunas de escore
`score`, `prediction`, `probability`, `pred`, `spai_score` ou `output`, e as
colunas de caminho `image`, `path`, `filename`, `file` ou `image_path`. Se
houver coluna de caminho, os escores são realinhados por nome de arquivo; caso
contrário, assume-se que a ordem foi preservada.

Saídas fora de `[0, 1]` são interpretadas como logits e convertidas por
sigmoide — transformação monotônica, que **não altera a AUC**.

**Se a CLI oficial mudar**, ajuste apenas o método `predict_proba` do adaptador.

### Pesos

O checkpoint é distribuído via Google Drive e exige confirmação manual.
Baixe-o conforme o README oficial e grave em `weights/spai.pth`.

---

## T04 — Geometria projetiva

**Repositório:** <https://github.com/hanlinm2/projective-geometry> (CVPR 2024)
**Componentes:** `perspective_fields/`, `line_segment/`, `object_shadow/`

### ⚠️ Bloqueio verificado: os extratores geométricos não acompanham o repositório

Os pesos distribuídos pelos autores são **apenas os classificadores finais**.
Nenhum dos três `Dataset` do repositório recebe imagens — todos carregam
representações geométricas **pré-computadas**:

| Componente | Entrada esperada (código oficial) | Extrator necessário |
|---|---|---|
| `perspective_fields` | `.pt` com `pred_latitude_original` e `pred_gravity_original` (`fields_dataset.py`) | [PerspectiveFields](https://github.com/jinlinyi/PerspectiveFields) — requer detectron2 |
| `line_segment` | dicionário `image_path_to_lines` com segmentos já detectados (`lines_dataset.py`) | detector de retas adotado pelos autores |
| `object_shadow` | máscaras PNG de objeto e de sombra (`dataset.py`) | [SSISv2](https://github.com/stevewongv/SSIS) — requer detectron2 + AdelaiDet |

Isso é consequência direta do desenho do método, não uma omissão: o artigo
afirma que *"all three classifiers are denied access to image pixels, and look
only at derived geometric features"*.

**Por que não substituir os extratores.** Trocar, por exemplo, o detector de
retas por `cv2.createLineSegmentDetector` produziria entradas fora da
distribuição sobre a qual os classificadores foram treinados. Os escores
resultantes não mediriam o método de Sarkar et al. (2024) e não constituiriam
replicação — configurando exatamente o risco R03 do Quadro 5. Uma substituição
só seria defensável se acompanhada de reajuste dos classificadores sobre a nova
representação, o que descaracterizaria o uso dos pesos oficiais.

**Barreira de ambiente.** detectron2 não tem suporte oficial em Windows e exige
compilação com MSVC. No ambiente deste trabalho não há `cl`, `nvcc` nem `conda`
disponíveis. Viabilizar T04 exigiria Linux (ou WSL2) com CUDA toolkit completo.

### ✅ Atualização: `object_shadow` deixou de estar bloqueado

O caminho previsto acima foi percorrido. Ubuntu 24.04 no WSL2, CUDA 12.1,
gcc-12, detectron2 recompilado com suporte a GPU — e o SSISv2 roda na RTX 3060.
O registro passo a passo, com os oito obstáculos vencidos (nenhum documentado
pelos projetos oficiais), está em [`../docs/T04_AMBIENTE_WSL2.md`](../docs/T04_AMBIENTE_WSL2.md).

A cadeia de `object_shadow` está completa:

| Etapa | Onde |
|---|---|
| Extração das máscaras de instância | `scripts/wsl/extrair_object_shadow_ssis.py` (no WSL2) |
| Agregação em mapas `Object` / `Shadow` | `agregar_por_classe`, no mesmo arquivo |
| Classificação com os pesos oficiais | `scripts/avaliar_t04_corpus.py` |

O adaptador não precisou do `infer.py` deste componente: consome os escores já
calculados. O contrato abaixo permanece válido para os **outros dois**.

**Situação atual:** um dos três componentes replicado ponta a ponta sobre o
corpus deste trabalho; `perspective_fields` e `line_segment` seguem sem
extrator. `is_available()` devolve disponibilidade **parcial**, e a agregação
por `nanmean` ignora as representações ausentes (RN07) — comportamento coberto
por teste automatizado. T04 como técnica completa segue não replicada.

---

### Contrato do adaptador (quando os extratores existirem)

O repositório oficial **não** expõe uma CLI unificada de inferência. O adaptador
espera, em cada diretório de componente, um script `infer.py` com esta assinatura:

```bash
python <componente>/infer.py \
    --input   <input.csv> \
    --output  <scores.json> \
    --weights <weights/projective_geometry/<componente>> \
    --device  cuda
```

- **Entrada:** CSV com coluna `image` (um caminho absoluto por linha).
- **Saída:** JSON no formato `{"<caminho da imagem>": <escore>, ...}`.
- O escore pode ser logit ou probabilidade; o adaptador normaliza por coluna.

### Como criar os scripts `infer.py`

Cada `infer.py` é um invólucro fino sobre o código de avaliação já presente no
repositório oficial. O trabalho consiste em: carregar o extrator geométrico e o
classificador daquele componente, iterar sobre os caminhos do CSV e gravar o
JSON. Esqueleto:

```python
import argparse, csv, json, torch

parser = argparse.ArgumentParser()
parser.add_argument("--input", required=True)
parser.add_argument("--output", required=True)
parser.add_argument("--weights", required=True)
parser.add_argument("--device", default="cuda")
args = parser.parse_args()

with open(args.input, newline="", encoding="utf-8") as handle:
    paths = [row["image"] for row in csv.DictReader(handle)]

model = load_official_model(args.weights, args.device)   # do repositório oficial

scores = {}
with torch.no_grad():
    for path in paths:
        representation = extract_geometry(path)           # do repositório oficial
        scores[path] = float(model(representation).squeeze())

with open(args.output, "w", encoding="utf-8") as handle:
    json.dump(scores, handle)
```

### Tolerância a falhas

Se um componente falhar ou estiver ausente, o adaptador registra `NaN` naquela
coluna e agrega apenas os componentes disponíveis (`nanmean`). A técnica só
falha por completo quando **as três** representações falham — comportamento
alinhado a RN07 e RNF04.

Os escores individuais das três representações ficam acessíveis em
`T04ProjectiveGeometry.last_component_scores_`, para a análise de erros da
Etapa 5.

---

## Verificação

Antes da campanha experimental, confirme a disponibilidade:

```bash
python -c "import sys; sys.path.insert(0,'.'); \
from src.techniques.t02_spai import T02SPAI; print(T02SPAI().is_available())"

python -c "import sys; sys.path.insert(0,'.'); \
from src.techniques.t04_geometry import T04ProjectiveGeometry; \
print(T04ProjectiveGeometry().is_available())"
```

Ambos retornam `(True, 'ok')` quando repositório, pesos e interpretador estão
no lugar. Caso contrário, a mensagem indica exatamente o que falta.

**Etapa 3 (verificação da implementação):** para T02 e T04, execute o pipeline
oficial sobre um subconjunto do benchmark e compare a AUC obtida à reportada
nos respectivos artigos. A implementação é considerada correta quando a
diferença permanece dentro de **cinco pontos percentuais** (risco R03).
