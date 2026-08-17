# Estado atual e como retomar

Atualizado em 16/08/2026.

Este arquivo existe para que o trabalho possa ser retomado em outra sessão sem
depender do histórico da conversa. Tudo que importa está no repositório.

---

## Como retomar em uma sessão nova

Basta apontar para este repositório e pedir a continuação. Um resumo do que
dizer:

> Estou continuando meu TCC 3 em `C:\Users\ferre\projects\tcc3-deteccao-imagens-ia`.
> Leia `docs/ESTADO_ATUAL.md` e continue do ponto de retomada. **Antes de supor
> qualquer progresso, confira o que ainda está vivo.**

**Situação em 17/08/2026: nada está rodando.** Os dois processos longos foram
parados de propósito, para liberar a máquina. A primeira ação de uma sessão nova
é **relançá-los** — os dois comandos estão na seção
[Como retomar](#️-como-retomar--os-dois-comandos-nesta-ordem).

Antes disso, confirme o estado, porque o log de uma execução interrompida para no
meio sem marca de erro:

```powershell
schtasks /query /tn "tcc3_t04_campos_test" /fo list | Select-String "Status"
Get-Content logs\t04_campos.log -Tail 3
Get-Content logs\pos_multiseed.log -Tail 3
Get-Process python | Select-Object Id,StartTime
```

"Em execução" significa vivo; "Pronto" significa terminado **ou parado** — e a
diferença só aparece no log e nos arquivos de saída.

O TCC 2 (documento de projeto) está em
`C:\Users\ferre\Downloads\TCC_2_Elian_Ferreira.pdf`.

As execuções longas rodam pelo Agendador de Tarefas do Windows e sobrevivem ao
fechamento do terminal — mas **não** ao fim da sessão; ver a armadilha 2.

```powershell
schtasks /query /tn "tcc3_pos_multiseed" /fo list
Get-Content logs\pos_multiseed.log -Tail 5
```

---

## Documentos do projeto

| Arquivo | Conteúdo |
|---|---|
| `docs/AUDITORIA_TCC2_TCC3.md` | O que o TCC 2 pediu × o que foi entregue, lacuna a lacuna |
| `docs/RESULTADOS.md` | Todos os resultados experimentais e sua interpretação |
| `docs/DECISOES_METODOLOGICAS.md` | Divergências em relação ao projeto, com justificativa |
| `docs/PLANO_ESCALA_INTEGRAL.md` | Viabilidade da escala e contingências; investigação do LSUN |
| `external/CONTRATO.md` | Integração com os repositórios oficiais; bloqueio de T04 |

Repositório: <https://github.com/elianferreira/tcc3-deteccao-imagens-ia> (privado)

---

## Em execução (seguem sozinhos, sem a sessão)

**Nada em execução.** Os dois processos foram **parados de propósito** em
17/08/2026, 07:20, para liberar a máquina. A GPU está livre (660 MB).

## ▶️ COMO RETOMAR — os dois comandos, nesta ordem

Retomar significa continuar de onde parou, e não recomeçar. O que já foi
computado está preservado; os comandos abaixo aproveitam isso.

### 1. Campos de perspectiva, split de teste (CPU, ~3 h)

```powershell
schtasks /run /tn "tcc3_t04_campos_test"
```

O extrator agora é **retomável por checkpoint**: grava o CSV a cada 250 imagens
e, ao reiniciar, pula o que já está lá. As 1.250 imagens processadas antes da
parada foram perdidas porque a gravação era só no fim — defeito corrigido, não
se repete.

### 2. Protocolo OOD por famílias (GPU, ~90 min)

**Não** relançar `tcc3_pos_multiseed`: o pipeline usa `--fit` e retreinaria tudo.
O treino **já terminou** e os modelos estão em `weights_ood_familias/`. Use:

```powershell
$env:TCC3_WEIGHTS_DIR = "C:\Users\ferre\projects\tcc3-deteccao-imagens-ia\weights_ood_familias"
.venv\Scripts\python.exe -u scripts\run_experiments.py --protocol ood `
  --manifest data\manifesto30k_ood_familias.csv `
  --techniques T01 T02 T03 T05 --refit-fusion
```

O que já está no cache (`results/scores/`) e será reaproveitado:

| Escore | Estado |
|---|---|
| T01, T02, T03 sobre `val` | ✅ prontos — inclui T02, que é o item caro |
| T01 sobre teste | ✅ pronto |
| T02, T03 sobre teste | ❌ recalcular (~80 min, quase todo de T02) |

Por isso ~90 min, e não os ~190 min de uma execução limpa.

**Atenção ao cache.** A chave é `{protocolo}__{técnica}__{condição}__{split}`, e
o protocolo é `ood` tanto para `manifesto30k_ood.csv` quanto para
`manifesto30k_ood_familias.csv` — os dois compartilham espaço de nomes. Não
causa erro porque as partições têm tamanhos diferentes e a guarda de forma em
`runner.py:119` rejeita o incompatível, mas a execução **sobrescreve** o cache
do OOD regular, que terá de ser recalculado se aquele protocolo for refeito.

### 3. Fila depois destes dois

1. `python scripts/consolidar_escores_t04.py` — junta as três representações em
   `results/t04_escores_componentes.csv`
2. `python scripts/avaliar_t04_componentes.py --split test` — a tabela de T04
   por representação e agregada
3. `python scripts/pipeline_t04_fusao.py` — T05 com T04 **completa**, que é a
   medição definitiva; a atual é provisória (ver 4.8.1 de `RESULTADOS.md`)

### Processos que morrem com a sessão

Em 15/08 à noite, `tcc3_pos_multiseed` e a regeneração do corpus pararam juntos
por volta das 23h55, sem traceback e com log de erro vazio. As tarefas do
Agendador estão registradas como **"Interativo apenas"**, de modo que terminam
junto com a sessão que as criou — o Agendador protege contra o fechamento do
shell, não contra o fim da sessão.

Ao retomar, sempre conferir o que de fato está vivo antes de assumir progresso:

```powershell
schtasks /query /tn "tcc3_t04_extracao" /fo list     # "Em execução" ou "Pronto"
Get-Process python | Select-Object Id,StartTime
```

## O que falta para o trabalho fechar

Em uma frase: **falta medir T05 com T04 completa.** Todo o resto está feito.

| # | Item | Estado |
|---|---|---|
| 1 | Campos de perspectiva no split de teste | rodando, ~3 h |
| 2 | Consolidar as três representações de T04 | script pronto, 1 comando |
| 3 | Tabela de T04 por representação | script pronto, 1 comando |
| 4 | **T05 com as quatro fontes completas** | **é o resultado que falta** |
| 5 | Protocolo OOD por famílias | rodando, fase de inferência |

O item 4 é o que responde à pergunta central da proposta. A medição atual de T05
(AUC 0,9996, peso −0,0403 para T04) é **provisória**: foi obtida quando T04
consistia apenas do componente objeto-sombra, um terço da técnica.

Previsão, a ser confirmada e não assumida: como as três representações medem
~0,53, a média deve ficar ~0,53 e T05 deve permanecer em 0,9996 com T04 perto de
peso zero. Mas há um caso em que isso falharia — se as três erram de formas
**diferentes**, a média pode ser menos ruidosa que cada uma isolada, e T04
receberia peso não desprezível. Isso seria resultado interessante por si só.

---

## PONTO DE RETOMADA — onde o trabalho parou

**A cadeia de T04 está fechada.** O que era o ponto de retomada anterior — o
agregador de máscaras — foi escrito, validado e executado ponta a ponta:

1. Agregador em `scripts/wsl/extrair_object_shadow_ssis.py`. O formato foi
   derivado de medição das máscaras oficiais, não arbitrado; ver
   [`T04_AMBIENTE_WSL2.md`](T04_AMBIENTE_WSL2.md), seção "O agregador".
   Coberto por `tests/test_t04_agregador.py`.
2. Extração sobre o corpus padrão: 59.999 pares nos três splits, **uma** falha
   (o defeito de pareamento do SSIS, obstáculo 8).
3. Avaliação com as três variantes de pesos oficiais —
   `scripts/avaliar_t04_corpus.py`.
4. T04 na tabela comparativa e como quarta fonte de T05 —
   `scripts/pipeline_t04_fusao.py`.

**O resultado é negativo, e essa é a contribuição.** T04 (objeto-sombra) fica em
acaso sobre este corpus: 0,5797 no melhor recorte (`outdoor`, pares com
conteúdo), contra 0,8216 do mesmo classificador no corpus de origem. Na fusão,
T05 permaneceu em 0,9996 e atribuiu a T04 peso −0,0403 — aprendeu a ignorá-la,
o que confirma por medição a tolerância prevista em RN07/RNF04.

Detalhes e as verificações que descartam erro de mapeamento de classes, falha do
extrator e o confundidor das máscaras vazias: seções 4.8 e 4.8.1 de
[`RESULTADOS.md`](RESULTADOS.md).

Ressalvas que precisam acompanhar o número no texto: é **um** dos três
componentes de T04, deve ser reportado como "T04 (objeto-sombra)", e é
transferência entre domínios sem reajuste, como a Etapa 2 exige.

### Próximo passo sugerido

Relançar `tcc3_pos_multiseed` (RNF01 com T02 + OOD por famílias), que a GPU
agora comporta. Depois, decidir a campanha em escala — ver "Decisão em aberto".

Verificação rápida de que o ambiente do WSL continua de pé:

```powershell
wsl -d Ubuntu-24.04 -u root -- /root/geo/bin/python -c "from detectron2 import _C; print(hasattr(_C,'modulated_deform_conv_forward'))"
```

Deve imprimir `True`. Se imprimir erro, refazer pelo `T04_AMBIENTE_WSL2.md`.

---

## Concluído

- Repositório publicado no GitHub
- Análise de erros (Etapa 5) — `scripts/analise_de_erros.py`
- Grad-CAM de T01 (Etapa 5) — `scripts/gradcam_t01.py`
- Replicação parcial de T04, componente objeto-sombra
- RNF01 em CPU para T01, T03 e T05
- Manifesto padrão do corpus em escala
- Corpus em escala montado: 192.638 imagens
- Três confundidores investigados: resolução, formato e perfil ICC
- **Multi-semente de T01 (Etapa 4)** — AUC 0,9964 ± 0,0002
- **Extratores de T04 desbloqueados no WSL2** — SSISv2 rodando na GPU
- **Corpus em escala regenerado sem perfil ICC** — 192.638 imagens conferidas
- **Manifestos da escala** — os três (padrão, OOD, OOD por famílias)
- **Cadeia de T04 fechada** — agregador, extração de 59.999 pares, avaliação
  com as três variantes de pesos e integração como quarta fonte de T05

### Resultados principais já obtidos

Protocolo padrão, corpus de 30k:

| Técnica | AUC | FPR |
|---|---|---|
| T01 | 0,9962 | 0,0233 |
| T02 | 0,9976 | 0,0504 |
| T03 | 0,8064 | 0,2371 |
| T04 (objeto-sombra) | 0,5384 | 0,1627 |
| T05 | 0,9996 | 0,0042 |

T04 entrou na tabela nesta sessão. Peso que T05 lhe atribui: −0,0403, contra
+4,25 de T02 e +3,41 de T01 — a fusão a ignora, e sua AUC não muda.

Protocolo OOD: colapso de 17 a 26 p.p., persistente com 60.000 imagens de
treino. Detalhes e testes de significância em `docs/RESULTADOS.md`.

Multi-semente de T01, conforme a Etapa 4 exige:

| Semente | AUC | Acurácia | Tempo |
|---|---|---|---|
| 42 | 0,9961 | 0,9689 | 102 min |
| 123 | 0,9965 | 0,9714 | 276 min |
| 456 | 0,9965 | 0,9713 | 132 min |
| **Média ± DP** | **0,9964 ± 0,0002** | **0,9706 ± 0,0014** | |

A semente 123 demorou mais por dividir CPU e disco com a montagem do corpus.

---

## Pendente

| # | Item | Observação |
|---|---|---|
| # | Item | Observação |
|---|---|---|
| 1 | Rodar as três representações de T04 sobre o corpus | extratores prontos; falta a GPU liberar |
| 2 | Isolar corpus × origem das máscaras em T04 | exige as imagens do Kandinsky; ver seção 4.8 de `RESULTADOS.md` |
| 3 | Campanha em escala | **decidido não executar** — ver "Decisão tomada" abaixo |

### Os três extratores de T04 estão de pé

O bloqueio de T04 nunca foi dos classificadores, e sim dos extratores. Os três
agora funcionam no WSL2:

| Representação | Extrator | Estado |
|---|---|---|
| `object_shadow` | SSISv2 | extraído: 59.999 pares, 3 splits |
| `perspective_fields` | PerspectiveFields (Jin et al., 2023) | pronto, testado |
| `line_segment` | DeepLSD (Pautrat et al., 2023) | em extração |

Os onze obstáculos vencidos estão em [`T04_AMBIENTE_WSL2.md`](T04_AMBIENTE_WSL2.md).

### Encerrados nesta sessão

| Item | Desfecho |
|---|---|
| Verificar T02 contra a AUC publicada | **feito** — `scripts/verificar_t02_publicado.py`, seção 4.9 de `RESULTADOS.md` |
| Conferir T04 contra a tabela do artigo | **impossível** — Sarkar et al. (2024) não publicam tabela de AUC, só curvas ROC |
| Manifestos da escala | **feito** — os três gerados |

---

## Decisão tomada: a campanha em escala não será executada

Decidido em 16/08/2026, depois de fechada a cadeia de T04.

A campanha sobre as 192.638 imagens levaria de 2,5 dias (semente única) a 4 dias
(três sementes). **Optou-se por não executá-la** e manter os resultados sobre o
corpus de 30k, que já são completos: protocolo padrão, OOD, OOD por famílias,
robustez, multi-semente e análise de erros.

Justificativa. O corpus em escala existe e está montado, com os três manifestos
prontos — a escala foi **viabilizada**, que era o objetivo da Etapa 1. O que a
execução acrescentaria é uma repetição das mesmas medições com mais dados, e a
Seção 3 de `DECISOES_METODOLOGICAS.md` já mostra que o colapso OOD persiste com
60.000 imagens de treino, isto é, não é efeito de escala insuficiente.

O tempo foi realocado para fechar mais um componente de T04, que é a lacuna real
em relação ao que o TCC 2 propôs.

**Reversível.** Nada foi descartado: corpus e manifestos estão em disco, e a
campanha é um comando (`scripts/pipeline_campanha.py --corpus
data/corvi2024_escala --prefixo escala`).

---

## Armadilhas conhecidas

Registradas porque cada uma já custou tempo nesta execução.

1. **Compile antes de rodada longa.** Um `SyntaxError` custou 12 h de janela.
   `python -m compileall scripts src app tests`.
2. **Processos longos morrem sem traceback.** Ocorreu cinco vezes, sempre com
   log de erro vazio. O Agendador (`scripts/multiseed.bat` como modelo) protege
   contra o fechamento do shell, mas **não** contra o fim da sessão: as tarefas
   estão em modo "Interativo apenas" e caíram junto com ela em 15/08, às 23h55.
   Ao retomar, verificar o que está vivo antes de supor progresso — o log para
   no meio, sem qualquer marca de erro.
3. **Não sobrescrever os pesos do protocolo padrão.** Treinos de outras
   variantes devem usar `TCC3_WEIGHTS_DIR`, como faz
   `scripts/pipeline_pos_multiseed.py`.
4. **T02 exige GPU livre.** Falha se houver treinamento em curso; não é defeito.
   A extração de T04 no WSL2 também ocupa a GPU — os dois não convivem na
   RTX 3060 de 6 GB.
5. **`wmic` não existe** nesta build do Windows 11; e a ferramenta PowerShell
   ficou indisponível durante a sessão. Bash funciona.
6. **A interface não é afetada** pelos treinos: `run_multi_seed` grava em
   `weights/t01_seed<N>.pt`, sem tocar em `t01_cooccurrence.pt`.
