# T04 — desbloqueio dos extratores geométricos no WSL2

Registro completo do ambiente que permitiu executar o SSISv2, extrator de
objeto-sombra de que T04 depende. Documentado passo a passo porque **nenhum dos
oito obstáculos encontrados está descrito na documentação oficial** dos
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

## Os oito obstáculos, na ordem em que aparecem

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

### 7. `postprocess` devolve `None`, e o `demo.py` oficial quebra nisso

Os dois últimos obstáculos só aparecem ao rodar sobre corpus real, não sobre as
amostras que os autores distribuem.

Ausência de detecção chega por **duas** rotas. A esperada é um `Instances`
vazio. A outra: `adet/modeling/ssis/condinst.py` inicializa
`final_results = None` em `postprocess` e só o atribui dentro do ramo
`if results.has("pred_global_masks")`. Quando nada sobrevive à filtragem, o
dicionário devolvido traz `{"instances": None}`.

O `demo.py` oficial faz `predictions["instances"].to(cpu)` sem verificar, e
portanto lança `AttributeError` nessas imagens. Não aparece na documentação
porque as amostras dos autores sempre têm alguma detecção; sobre o corpus deste
trabalho, ocorreu já na primeira centena.

As duas rotas significam a mesma coisa — nenhum objeto com sombra projetada — e
o extrator as trata igual, com um par de mapas pretos, registrando em qual
delas cada imagem caiu.

### 8. Defeito no pareamento do SSIS derruba a rodada inteira

O mais caro dos oito, porque só se manifesta depois de horas de execução.

`adet/modeling/ssis/condinst.py`, na rotina que casa objeto com sombra:

```python
record.pop(record[ind])   # linha 321
record.pop(ind)           # linha 322  -> KeyError
```

`record` guarda o pareamento nos dois sentidos (`record[ind] = i` e
`record[i] = ind`). Quando as duas entradas coincidem, a primeira remoção já
apaga a chave que a segunda tenta remover, e a inferência morre com
`KeyError: np.int64(12)`.

Frequência: **uma imagem em 42.000**. Bastou para perder 15.358 imagens de
progresso no split de treino, porque a exceção sobe até o topo do laço.

Correção adotada: capturar por imagem no extrator, registrar em
`tcc3_30k_falhas_<split>_<carimbo>.csv` e seguir. A imagem fica sem mapas e não
entra na avaliação — a exclusão é rara e auditável, o que é preferível a alterar
o código oficial dos autores.

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
uma de sombra, uma de objeto — conforme `object_shadow/dataset.py:20-21`.

---

## O agregador

`scripts/wsl/extrair_object_shadow_ssis.py`, função `agregar_por_classe`.

O formato de saída não foi arbitrado: as máscaras publicadas pelos autores
(`Kandinsky_Outdoor_{shadow,object}`) foram medidas antes, porque o
classificador foi treinado nelas. Amostra de 200 arquivos por classe:

| Propriedade | Valor medido |
|---|---|
| Modo e tamanho | `L`, 256×256, uint8 |
| Massa em [20, 235] | **exatamente 0** |
| Formato em disco | JPEG |

O intervalo vazio é o achado que decide o agregador: os mapas são estritamente
binários, sem nível de cinza reservado a distinguir instâncias. Logo a redução
das `N` máscaras é a **união binária** por classe — qualquer codificação mais
rica divergiria do que o classificador viu no treino. Os valores 1–6 e 249–253
que aparecem nos arquivos são artefato de JPEG, não sinal.

Os mapas produzidos aqui foram conferidos contra esse mesmo critério e também
têm massa zero em [20, 235]. A cobertura está em `tests/test_t04_agregador.py`,
que roda no Windows sem detectron2 — o agregador é função pura sobre arrays.

## O que falta

1. ~~Agregador de máscaras por classe~~ — feito
2. Extração sobre o corpus, em andamento: 0,21 s por imagem, ~3,5 h para as
   60.000 do manifesto padrão (`scripts/t04_extracao.bat`, retomável)
3. Avaliação pelo classificador oficial — `scripts/avaliar_t04_corpus.py`
4. T04 na tabela comparativa e como quarta fonte da fusão T05

Os outros dois componentes de T04 — campos de perspectiva e segmentos de reta —
seguem sem extrator. O caminho aberto aqui os torna plausíveis, não resolvidos.
