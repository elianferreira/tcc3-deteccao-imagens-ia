# T04 — desbloqueio dos extratores geométricos no WSL2

Registro completo do ambiente que permitiu executar o SSISv2, extrator de
objeto-sombra de que T04 depende. Documentado passo a passo porque **nenhum dos
seis obstáculos encontrados está descrito na documentação oficial** dos
projetos envolvidos, e refazer sem este registro custaria as mesmas horas.

Situação anterior: T04 constava como bloqueada porque detectron2 não tem
suporte oficial em Windows. Isso continua verdadeiro — a solução é executar em
Linux, via WSL2, o que a Seção 3.2 do TCC 2 já contempla ao admitir ambiente
alternativo.

**Importante:** nada em T04 é treinado. São usados os pesos oficiais dos
autores, tanto do SSISv2 quanto dos classificadores, em conformidade com a
Etapa 2 do TCC 2.

---

## Ambiente

| Item | Versão |
|---|---|
| Distribuição | Ubuntu 24.04 LTS no WSL2 |
| Python | 3.12.3 (venv em `/root/geo`) |
| PyTorch | 2.4.1+cu121 |
| CUDA toolkit | 12.1 (`/usr/local/cuda-12.1`) |
| Compilador | **gcc-12** (não o padrão 13) |
| detectron2 | 0.6, compilado com CUDA |
| GPU | RTX 3060 6 GB, via `/dev/dxg` |

A passagem de GPU para o WSL2 funciona sem configuração adicional: `nvidia-smi`
já enxerga a placa, pois o driver vem do Windows.

---

## Os seis obstáculos, na ordem em que aparecem

### 1. `pip install -e .` do detectron2 falha com `No module named 'torch'`

O `setup.py` importa torch em tempo de compilação, mas o pip isola o ambiente
de build.

```bash
pip install -e . --no-build-isolation
```

### 2. `adet` (SSIS) exige `kornia==0.5.6`

Versão de 2021, mas instala e funciona com torch 2.4. Emite apenas um
`FutureWarning` sobre decoração de classes, inofensivo.

### 3. `pysobatools` não compila: `longintrepr.h: No such file or directory`

O `_mask.c` distribuído foi gerado por um Cython antigo, mirando Python ≤ 3.10;
esse cabeçalho passou a ser interno no Python 3.11. Solução: regerar do `.pyx`.

```bash
pip install cython
cd InstanceShadowDetection/PythonAPI
rm -f pysobatools/_mask.c
cython pysobatools/_mask.pyx
```

### 4. Após regerar: `../common/maskApi.c: No such file or directory`

O `_mask.pyx` traz `# distutils: sources = ../common/maskApi.c`, caminho
relativo que não resolve. **Remover a linha** — o `setup.py` já declara
`./common/maskApi.c` corretamente. Mantê-la e apenas corrigir o caminho causa o
obstáculo 5.

```bash
sed -i "/# distutils: sources =/d" pysobatools/_mask.pyx
rm -rf build pysobatools/_mask.c pysobatools/*.so
cython pysobatools/_mask.pyx
python setup.py build_ext --inplace
```

### 5. `multiple definition of 'rleToString'`

Ocorre se a diretiva do `.pyx` e o `setup.py` declararem a mesma fonte: o
`maskApi.c` entra duas vezes na ligação. Resolvido pelo passo anterior.

### 6. `Detectron2 is not compiled with GPU support!`

Sem `nvcc` no ambiente, o detectron2 compila apenas para CPU — e CPU **não
serve**: `Deformable Conv is not supported on CPUs!`. É preciso o toolkit CUDA
e recompilar.

```bash
# repositorio da NVIDIA para WSL
wget https://developer.download.nvidia.com/compute/cuda/repos/wsl-ubuntu/x86_64/cuda-keyring_1.1-1_all.deb
dpkg -i cuda-keyring_1.1-1_all.deb && apt-get update

# cuda-toolkit-12-1 completo falha com pacotes quebrados; instalar por partes
apt-get install -y cuda-nvcc-12-1 cuda-cudart-dev-12-1 \
    libcusparse-dev-12-1 libcublas-dev-12-1 libcusolver-dev-12-1 \
    libcurand-dev-12-1 libcufft-dev-12-1

# gcc 13 e rejeitado pelo CUDA 12.1: "gcc versions later than 12 are not supported"
apt-get install -y gcc-12 g++-12
```

Recompilação com as variáveis corretas:

```bash
cd /root/detectron2
export CUDA_HOME=/usr/local/cuda-12.1
export PATH="$CUDA_HOME/bin:$PATH"
export TORCH_CUDA_ARCH_LIST="8.6"     # RTX 3060 = compute capability 8.6
export CC=gcc-12 CXX=g++-12
export NVCC_PREPEND_FLAGS="-ccbin /usr/bin/g++-12"
rm -rf build
/root/geo/bin/pip install -e . --no-build-isolation
```

Verificação:

```python
from detectron2 import _C
assert hasattr(_C, "modulated_deform_conv_forward")
```

---

## Pesos e dados auxiliares

Baixados do Google Drive oficial do SSIS com `gdown --folder`:

| Arquivo | Tamanho | Uso |
|---|---|---|
| `model_ssisv2_final.pth` | 587 MB | o extrator usado |
| `SOBA_v2.zip` | 175 MB | anotações, só para registrar metadados |

O `SOBA_val_v2.json` precisa existir em `../../dataset/SOBA/annotations/`
relativo a `demo/`, ainda que a inferência não use o dataset: o `demo.py`
registra o catálogo na importação.

Os pesos vão para
`tools/output/SSISv2_MS_R_101_bifpn_with_offset_class_maskiouv2_da_bl/`.

---

## Execução verificada

```bash
cd /root/SSIS/demo
/root/geo/bin/python demo.py \
  --config-file ../configs/SSIS/MS_R_101_BiFPN_SSISv2_demo.yaml \
  --input <diretorio> --output <diretorio> \
  --opts MODEL.WEIGHTS ../tools/output/SSISv2_.../model_ssisv2_final.pth
```

`--input` aceita **diretório**, não arquivo nem curinga.

Resultado sobre imagens do corpus deste trabalho:

```
coco_000000.png ............ 0 instâncias em 0,87 s
latent_diffusion_000000.png  4 instâncias em 0,17 s
```

## Formato da saída

```
campos:       pred_masks, pred_classes, pred_boxes, scores,
              pred_associations, locations, fpn_levels, offset
thing_classes: ['Object', 'Shadow']     # classe 0 = objeto, 1 = sombra
pred_masks:   (N, 256, 256) float32
```

O classificador de T04 consome **duas** imagens em tons de cinza por amostra —
uma de sombra, uma de objeto — conforme `object_shadow/dataset.py:20-21`. Falta
escrever o agregador que reduz as N máscaras de instância a esses dois mapas
binários.

---

## O que falta

1. Escrever o agregador de máscaras por classe (`Object` / `Shadow`)
2. Extrair sobre o corpus deste trabalho — ~0,17 s por imagem na GPU, cerca de
   40 min para as 12.638 do benchmark
3. Alimentar os classificadores oficiais, já validados em
   `scripts/replicar_t04_object_shadow.py`
4. Só então T04 entra na tabela comparativa e na fusão T05

Os outros dois componentes de T04 — campos de perspectiva e segmentos de reta —
seguem sem extrator. O caminho aberto aqui os torna plausíveis, não resolvidos.
