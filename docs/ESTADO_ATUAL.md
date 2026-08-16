# Estado atual e como retomar

Atualizado em 15/08/2026, 23:15.

Este arquivo existe para que o trabalho possa ser retomado em outra sessão sem
depender do histórico da conversa. Tudo que importa está no repositório.

---

## Como retomar em uma sessão nova

Basta apontar para este repositório e pedir a continuação. Um resumo do que
dizer:

> Estou continuando meu TCC 3 em `C:\Users\ferre\projects\tcc3-deteccao-imagens-ia`.
> Leia `docs/ESTADO_ATUAL.md`, `docs/AUDITORIA_TCC2_TCC3.md` e
> `docs/RESULTADOS.md` para se situar, e continue de onde parou.

O TCC 2 (documento de projeto) está em
`C:\Users\ferre\Downloads\TCC_2_Elian_Ferreira.pdf`.

**Nada em execução depende da sessão.** As duas execuções longas rodam pelo
Agendador de Tarefas do Windows (`tcc3_multiseed` e `tcc3_pos_multiseed`) e
continuam mesmo com o terminal fechado.

```powershell
schtasks /query /tn "tcc3_multiseed" /fo list
schtasks /query /tn "tcc3_pos_multiseed" /fo list
Get-Content logs\multiseed.log -Tail 5
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

**Nada em execução.** A GPU está livre.

O próximo item da fila é `tcc3_pos_multiseed`, que parou em 15/08 e precisa ser
relançado do zero (RNF01 com T02 e o protocolo OOD por famílias):

```powershell
schtasks /run /tn "tcc3_pos_multiseed"
```

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
| 1 | RNF01 com T02 + OOD por famílias | `tcc3_pos_multiseed`; GPU livre, pode ser relançado |
| 2 | Isolar corpus × origem das máscaras em T04 | exige o dataset Kandinsky dos autores; ver seção 4.8 de `RESULTADOS.md` |
| 3 | Campanha em escala | `scripts/pipeline_campanha.py --corpus data/corvi2024_escala --prefixo escala`; ~4 dias com três sementes |
| 4 | Verificar T02 contra a AUC publicada | Etapa 3, tolerância de 5 p.p. |
| 5 | Conferir T04 contra a **tabela** do artigo | os valores usados vieram da figura |
| 6 | T04 completa | perspectiva e segmentos de reta seguem sem extrator; ver `external/CONTRATO.md` |

---

## Decisão em aberto

A campanha em escala com as três sementes leva cerca de 4 dias; com semente
única, 2 dias e meio. A Etapa 4 exige as três explicitamente, mas a
variabilidade medida na escala de 30k foi de 0,0004 em AUC entre as sementes 42
e 123.

Alternativa defensável: rodar a escala com semente única e reportar o desvio
padrão medido em 30k, declarando a escolha. **Não decidido.**

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
