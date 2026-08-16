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

## Em execução

| Item | Mecanismo | Como acompanhar |
|---|---|---|
| Multi-semente de T01 (semente 456) | tarefa `tcc3_multiseed` | `logs/multiseed.log` |
| RNF01 com T02 + OOD por famílias | tarefa `tcc3_pos_multiseed` | `logs/pos_multiseed.log` |
| Regeneração das reais sem perfil ICC | processo destacado | `logs/corpus_reparo_icc.log` |

O multi-semente é **retomável**: cada semente concluída fica em
`results/multiseed/` e é reaproveitada se a execução for reiniciada.

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

### Resultados principais já obtidos

Protocolo padrão, corpus de 30k:

| Técnica | AUC | FPR |
|---|---|---|
| T01 | 0,9962 | 0,0233 |
| T02 | 0,9976 | 0,0504 |
| T03 | 0,8064 | 0,2371 |
| T05 | 0,9996 | 0,0042 |

Protocolo OOD: colapso de 17 a 26 p.p., persistente com 60.000 imagens de
treino. Detalhes e testes de significância em `docs/RESULTADOS.md`.

Multi-semente (parcial): semente 42 com AUC 0,9961; semente 123 com 0,9965.

---

## Pendente

| # | Item | Observação |
|---|---|---|
| 1 | Manifestos OOD e OOD-famílias da escala | só o padrão foi gerado |
| 2 | Campanha em escala | `scripts/pipeline_campanha.py --corpus data/corvi2024_escala --prefixo escala`; ~4 dias com três sementes |
| 3 | Verificar T02 contra a AUC publicada | Etapa 3, tolerância de 5 p.p. |
| 4 | Conferir T04 contra a **tabela** do artigo | os valores usados vieram da figura |
| 5 | T04 completa | bloqueada em definitivo; ver `external/CONTRATO.md` |

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
2. **Processos lançados pelo shell morrem.** Ocorreu três vezes, inclusive com
   `DETACHED_PROCESS`, sempre sem traceback e com log de erro vazio. Use o
   Agendador (`scripts/multiseed.bat` como modelo) para execuções longas.
3. **Não sobrescrever os pesos do protocolo padrão.** Treinos de outras
   variantes devem usar `TCC3_WEIGHTS_DIR`, como faz
   `scripts/pipeline_pos_multiseed.py`.
4. **T02 exige GPU livre.** Falha se houver treinamento em curso; não é defeito.
5. **`wmic` não existe** nesta build do Windows 11; e a ferramenta PowerShell
   ficou indisponível durante a sessão. Bash funciona.
6. **A interface não é afetada** pelos treinos: `run_multi_seed` grava em
   `weights/t01_seed<N>.pt`, sem tocar em `t01_cooccurrence.pt`.
