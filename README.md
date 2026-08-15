# Detecção de Imagens Geradas por Inteligência Artificial

Implementação do TCC 3 de **Elian Ferreira** — Ciência da Computação, UNIVALI, 2026.
Orientador: MSc. Felipe Viel.

Replicação e avaliação comparativa de quatro técnicas forenses de detecção de
imagens sintéticas, sob um protocolo experimental comum, mais uma quinta técnica
de fusão proposta neste trabalho.

| ID | Técnica | Domínio de evidência | Implementação |
|----|---------|----------------------|---------------|
| **T01** | Coocorrência RGB + CNN — [Nataraj et al., 2019](https://arxiv.org/abs/1903.06836) | Espacial | Replicação treinada localmente |
| **T02** | SPAI — [Karageorgiou et al., CVPR 2025](https://github.com/mever-team/spai) | Espectral | Código e pesos oficiais (inferência) |
| **T03** | Lei de Benford sobre DCT — [Bonettini et al., 2020](https://arxiv.org/abs/2004.07682) | Estatístico | Replicação treinada localmente |
| **T04** | Geometria projetiva — [Sarkar et al., CVPR 2024](https://github.com/hanlinm2/projective-geometry) | Geométrico | Extratores e código oficiais |
| **T05** | **Arquitetura híbrida (proposta)** | Fusão tardia dos quatro | Contribuição original |

Métrica primária: **AUC-ROC**. Protocolos: **padrão (in-distribution)**,
**generalização (OOD)** e **robustez**.

---

## Instalação

```bash
git clone <seu-repositório>
cd tcc3-deteccao-imagens-ia

python -m venv .venv
.venv\Scripts\activate            # Windows
# source .venv/bin/activate       # Linux/macOS

pip install -r requirements.txt
```

PyTorch com suporte a GPU (RTX 3060, 6 GB):

```bash
pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu124
```

> **Certificado SSL:** se o `pip` falhar com `CERTIFICATE_VERIFY_FAILED`
> (comum atrás de antivírus que inspeciona TLS), acrescente:
> `--trusted-host pypi.org --trusted-host files.pythonhosted.org --trusted-host download.pytorch.org`

### Repositórios oficiais de T02 e T04

```bash
python scripts/setup_external.py --create-venvs
```

T02 e T04 rodam em **subprocesso**, sobre ambientes virtuais próprios: o SPAI
exige Python 3.11 + CUDA 12.4, incompatível com o ambiente principal. Os pesos
pré-treinados exigem download manual — o script imprime os endereços ao final.
O contrato de integração está em [`external/CONTRATO.md`](external/CONTRATO.md).

---

## Uso

### 1. Validar o código sem o dataset completo

O dataset de Corvi et al. tem 360.000 imagens. Para exercitar todo o fluxo antes
de baixá-lo:

```bash
python scripts/make_sample_corpus.py --output data/amostra --per-class 200
python scripts/prepare_dataset.py --corpus data/amostra --expected-size 256
python scripts/run_experiments.py --protocol standard --techniques T01 T03 T05 --fit
```

> As métricas obtidas sobre o corpus de amostra **não têm valor científico** e
> não devem ser reportadas na monografia.

### 2. Preparar o dataset real (Etapa 1)

Organize o corpus assim:

```
data/corvi2024/
    real/{raise,fodb,imagenet,coco,open_images}/*.png
    fake/{glide,stable_diffusion_1_3,...,midjourney_v6_1}/*.png
```

```bash
# Protocolo padrão: particionamento estratificado 70/15/15
python scripts/prepare_dataset.py --corpus data/corvi2024

# Protocolo OOD: treino em difusão, teste em GigaGAN e Midjourney
python scripts/prepare_dataset.py --corpus data/corvi2024 --protocol ood
```

A verificação de integridade confere contagem por classe e por gerador,
resolução, formato, arquivos corrompidos e duplicatas entre partições. O
relatório vai para `data/integridade_<protocolo>.json`.

### 3. Campanha experimental (Etapa 4)

```bash
# Protocolo padrão, treinando T01, T03 e T05
python scripts/run_experiments.py --protocol standard --fit

# Generalização a geradores não vistos
python scripts/run_experiments.py --protocol ood --manifest data/manifesto_ood.csv --fit

# Robustez: 9 perturbações sobre subamostra do teste
python scripts/run_experiments.py --protocol robustness --limit 2000

# Três inicializações de T01 (sementes 42, 123 e 456)
python scripts/run_experiments.py --protocol standard --multi-seed
```

Saídas em `results/`: métricas agregadas e por gerador em CSV, probabilidades
brutas em `.npy` (para curvas ROC, teste de DeLong e análise de erros sem
repetir a inferência) e figuras em PDF/PNG.

### 4. Interface demonstrativa (Etapa 6)

```bash
python app/gradio_app.py
```

Aceita PNG, JPEG e WebP até 10 MB. Exibe o escore de cada técnica em percentual
e o espectro de magnitude. **Não emite classificação binária automática** — a
interpretação cabe ao usuário (RN04).

### 5. Testes (Seção 3.6.2)

```bash
pytest                              # todos
pytest tests/test_unit_features.py  # unitários
pytest tests/test_interface.py      # interface e regras de negócio
```

---

## Estrutura

```
src/
  config.py            constantes experimentais e hiperparâmetros
  seeds.py             determinismo (RNF02)
  metrics.py           AUC, acurácia, F1, FPR/FNR, DeLong, bootstrap
  plots.py             curvas ROC, AUC por gerador, heatmap de robustez
  data/
    manifest.py        indexação, verificação de integridade, particionamento
    perturbations.py   grade do protocolo de robustez
  techniques/
    base.py            interface fit / predict_proba / score
    t01_..t05_*.py     as cinco técnicas
  experiments/
    runner.py          orquestração dos três protocolos
app/gradio_app.py      interface demonstrativa
scripts/               preparação, execução e setup dos repositórios externos
tests/                 unitários, integração, interface, desempenho
```

### Decisões de projeto que valem registro na monografia

> **Leia [`docs/DECISOES_METODOLOGICAS.md`](docs/DECISOES_METODOLOGICAS.md).**
> Reúne as alterações metodológicas introduzidas durante a execução que **não**
> constavam do projeto aprovado no TCC 2, cada uma com o motivo, a evidência
> empírica que a exigiu e o impacto sobre a leitura dos resultados. Todas
> precisam ser declaradas no Capítulo 4.

**T05 nunca é ajustada sobre o conjunto de treinamento.** T01 e T03 produzem
escores otimistas sobre os dados em que foram ajustadas; calibrar a fusão neles
levaria T05 a subestimar o peso das técnicas treinadas localmente. Há duas
partições válidas para a calibração, e a escolha entre elas mudou o resultado:

| `--fusion-split` | Partição | Quando usar |
|---|---|---|
| `val` | validação in-distribution | reproduz o projeto original do TCC 2 |
| `fusion` | gerador *held-out* (`glide`) + reais reservadas | protocolo OOD |

A calibração `val` produziu **T05 = 0,5880 no OOD, abaixo de T01 isolada
(0,6369)** — a fusão piorou o resultado. O motivo: na validação as três fontes
acertam quase tudo, então a fusão aprende pesos válidos apenas para um regime
em que todas concordam, regime que não se repete diante de geradores não vistos.
A partição `fusion` corrige isso calibrando sobre um gerador que exibe a mesma
degradação do conjunto de avaliação.

**Ambas devem ser reportadas.** A comparação entre elas é um resultado do
trabalho, não um detalhe de ajuste.

**T05 recebe uma máscara de disponibilidade além dos quatro escores.**
Permite ao classificador distinguir "escore 0,5 observado" de "escore ausente
porque o módulo falhou", condição para que a fusão continue operando quando uma
técnica está indisponível (RN07, RNF04).

**Avaliação por gerador usa todas as imagens reais.** Cada gerador é avaliado
contra o conjunto completo de reais, de modo que a AUC seja definida mesmo quando
o subconjunto sintético é de um único gerador.

**Escores fora de [0, 1] são convertidos por sigmoide.** Transformação
monotônica: preserva a ordenação e, portanto, não altera a AUC.

---

## Reprodutibilidade (RNF02)

Sementes fixadas em 42 (`random`, NumPy, PyTorch, CUDA), modo determinístico
habilitado, `CUBLAS_WORKSPACE_CONFIG=:4096:8`, versões travadas em
`requirements.txt`. A mesma imagem produz escores idênticos em execuções
distintas — verificado por teste automatizado.

## Pesos versionados neste repositório

| Arquivo | Situação |
|---|---|
| `weights/t01_cooccurrence.pt` | ✓ versionado (2,8 MB) |
| `weights/t05_fusion.pkl` | ✓ versionado, variantes A e B |
| `weights/t03_benford.pkl` | ✗ **não versionado** — 337 MB, acima do limite de 100 MB por arquivo do GitHub |
| `weights/spai.pth` | ✗ artefato oficial de terceiros (891 MB) |
| `weights/projective_geometry/` | ✗ artefato oficial de terceiros |

O modelo de T03 é integralmente regenerável:

```powershell
python scripts/run_experiments.py --protocol standard --fit --techniques T03
```

Os pesos oficiais de T02 e T04 são obtidos por `scripts/setup_external.py`;
redistribuí-los aqui seria indevido.

## Ambiente

Intel Core i7 12ª geração, 16 GB RAM, NVIDIA RTX 3060 6 GB VRAM.
Suporta o treinamento de T01, T03 e T05 e a inferência de T02 e T04; **não** o
treinamento do SPAI, que exige GPU de 48 GB. Google Colab como ambiente
alternativo se necessário.

## Licença

Código deste trabalho: MIT. T02 segue Apache 2.0 (mever-team/spai); T04 segue a
licença de hanlinm2/projective-geometry. Os datasets mantêm as licenças originais.
