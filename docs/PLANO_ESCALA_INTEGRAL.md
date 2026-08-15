# Escala integral do TCC 2 — viabilidade e plano de contingência

O TCC 2 (Seção 3.5.1) especifica **180.000 imagens reais + 180.000 sintéticas =
360.000**, proporção 1:1, todas em 256 × 256. Este documento confronta essa
meta com o que é efetivamente possível na máquina do projeto e define o plano
de execução e as contingências.

---

## 1. Inventário verificado

Medições feitas em 15/08/2026, não estimativas.

### Sintéticas — meta atingível ✓

```
latent_diffusion_trainingset.zip
  200.000 imagens no total
  180.000 em /train/        <- exatamente a meta do TCC
```

### Reais — meta **não** atingível ✗

O conjunto real oficial de Corvi et al. (2024) é definido por duas listas
dentro do próprio arquivo de treino:

| Lista oficial | Nomes | Fonte | Disponível |
|---|---|---|---|
| `train/real_coco.txt` | 90.000 | COCO train2017 | ✓ 118.287 imagens baixadas |
| `train/real_lsun.txt` | 90.000 | LSUN | ✗ **não baixado** |

O teto real hoje é de **90.000 imagens reais**, metade da meta.

O TCC 2 cita cinco bases reais (RAISE, FODB, ImageNet, COCO, Open Images).
Dessas, apenas COCO está disponível: RAISE exige formulário de registro,
ImageNet exige conta aprovada, FODB e Open Images exigem cadastro. LSUN, que é
a fonte efetivamente usada por Corvi, é distribuída em formato LMDB por
categoria, com dezenas de gigabytes por categoria.

### Armazenamento — bloqueio imediato ✗

```
Disco C:    475 GB total    444 GB usados    32 GB livres    (94%)
Tamanho médio de imagem normalizada: 114,6 KB
```

| Cenário | Imagens | Espaço só do corpus normalizado |
|---|---|---|
| Meta do TCC | 360.000 | **41,3 GB** |
| Máximo sem LSUN | 180.000 | **20,6 GB** |
| Atual | 72.638 | 8,0 GB |

Com 32 GB livres, a meta do TCC **não cabe** — e isso ignora a cópia
intermediária não normalizada, que o fluxo atual também grava.

---

## 2. Plano adotado: 90.000 + 90.000 = 180.000 imagens

É o maior conjunto **balanceado 1:1** possível sem LSUN, e usa exatamente as
90.000 imagens reais da lista oficial de Corvi — ou seja, é fiel ao trabalho
original naquilo que está ao alcance, e não uma amostra arbitrária.

Representa **2,5×** o corpus atual e metade da meta do TCC.

### Contingência A — eliminar a cópia intermediária

O fluxo atual grava duas vezes: `organize_corpus.py` extrai para
`data/<corpus>/` e `normalize_corpus.py` grava em `data/<corpus>_norm/`. Para
180.000 imagens isso custaria ~24 GB desnecessários.

**Ação:** extrair e normalizar em passagem única, gravando apenas a saída
normalizada. Reduz a necessidade de ~45 GB para ~21 GB.

### Contingência B — liberar espaço

Corpora invalidados que podem ser removidos com segurança:

| Diretório | Motivo |
|---|---|
| `data/corvi2024/` | corpus de 22.638 com confundidor, invalidado |
| `data/corvi2024_norm/` | normalização daquele corpus, invalidada |
| `data/corvi2024_30k/` | intermediário não normalizado; regenerável dos zips |
| `data/amostra/` | corpus sintético de testes |

Os arquivos `.zip` em `data/downloads/` (~62 GB) **não** devem ser apagados:
são a fonte de qualquer reprocessamento e o download do COCO levou horas.

### Contingência C — se o espaço ainda faltar

1. Reduzir para 60.000 + 60.000 (13,8 GB), ainda 1,65× o corpus atual
2. Mover `data/downloads/` para disco externo, liberando 62 GB
3. Manter apenas o corpus normalizado e regenerar o resto sob demanda

---

## 3. Custo de tempo estimado

Extrapolado da rodada de 30k (19 h para 72.638 imagens).

| Etapa | 72.638 (medido) | 180.000 (estimado) |
|---|---|---|
| Normalização | ~7 h | ~17 h |
| Treino T01, 1 semente | ~4 h | ~10 h |
| Treino T01, 3 sementes (Etapa 4) | — | ~30 h |
| Busca em grade de T03 | ~6 h | ~15 h |
| Inferência T02 (434 ms/imagem) | ~1,5 h | ~3,5 h |
| Protocolos OOD e robustez | ~7 h | ~15 h |
| **Total** | **19 h** | **~90 h (≈ 4 dias)** |

O gargalo é o treino com três sementes, exigido pela Etapa 4. Sem ele o total
cai para ~70 h.

---

## 4. Riscos e resposta

| Risco | Probabilidade | Resposta |
|---|---|---|
| Disco esgota durante a normalização | Alta | Verificação de espaço livre antes de cada etapa, com parada limpa |
| Interrupção da máquina em 4 dias de execução | Alta | Já mitigado: normalização e manifestos são idempotentes e retomáveis |
| T03 estoura RAM com 150.000 amostras × 540 características | **Baixa** (reavaliado) | Ver nota abaixo |
| Modelo de T03 acima de 1 GB | Alta | Já ocorre: 337 MB com 60.000 amostras. Não versionável; regenerável |
| Resultados não melhorarem com a escala | Média | É resultado válido: indicaria saturação, e o colapso OOD passaria a ser atribuível à técnica, não ao volume de dados |

### Nota sobre a memória de T03 — risco reavaliado para baixo

A avaliação inicial deste documento afirmava que a busca em grade replicaria a
matriz de características por processo, elevando o risco de estouro de RAM.
**Isso está incorreto e foi verificado no código.**

`src/techniques/t03_benford.py:297` já configura `GridSearchCV(n_jobs=1)`, com
o comentário explícito de que "o paralelismo já ocorre dentro da floresta"
(`RandomForestClassifier(n_jobs=-1)`). Como o joblib não cria processos para os
pontos da grade, a matriz **não** é duplicada.

Consumo real esperado a 150.000 amostras:

| Item | Tamanho |
|---|---|
| Matriz completa (150.000 × 540 × 8 bytes) | 648 MB |
| Partição de treino da validação cruzada (80%) | 518 MB |
| Floresta ajustada | ~1 GB (extrapolado dos 337 MB a 60.000) |

Total da ordem de 2 a 3 GB, confortável nos 16 GB disponíveis. O risco
remanescente é o **tamanho do modelo em disco**, não a memória.

Contingência caso ainda assim falte memória: reduzir `grid_n_estimators` de
`(100, 200, 500)` para `(100, 200)`. Isso altera a metodologia e precisaria ser
declarado, por isso não foi aplicado preventivamente.

---

## 5. O que fica declarado na monografia

Independentemente do resultado, a Seção 4 deve declarar:

1. O corpus tem **metade** das imagens reais previstas, por indisponibilidade
   de LSUN e das demais quatro bases.
2. As 90.000 reais são **exatamente** as da lista oficial de Corvi et al.
   (2024), não uma amostra arbitrária — a redução é de quantidade, não de
   representatividade da fonte.
3. A proporção 1:1 entre classes, exigida pela Seção 3.5.1, **é mantida**.
