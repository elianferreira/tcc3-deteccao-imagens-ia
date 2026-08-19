# Estado atual e como retomar

Atualizado em 19/08/2026.

Este arquivo existe para que o trabalho possa ser retomado em outra sessão sem
depender do histórico da conversa. Tudo que importa está no repositório.

---

## Como retomar em uma sessão nova

> Estou continuando meu TCC 3 em `C:\Users\ferre\projects\tcc3-deteccao-imagens-ia`.
> Leia `docs/ESTADO_ATUAL.md` e continue do ponto de retomada. **Antes de supor
> qualquer progresso, confira o que ainda está vivo.**

**Em 19/08/2026 não havia nada em execução** e nenhuma etapa pendente. Se uma
sessão futura deixar algo rodando, confirme antes de supor progresso — o log de
uma execução interrompida para no meio sem marca de erro:

```powershell
Get-Process python | Select-Object Id,StartTime
schtasks /query /fo list | Select-String "tcc3_"
```

"Em execução" significa vivo. "Pronto" significa terminado **ou morto** — e a
diferença só aparece no log e nos arquivos de saída.

O TCC 2 (documento de projeto) está em
`C:\Users\ferre\Downloads\TCC_2_Elian_Ferreira.pdf`.

---

## Documentos do projeto

| Arquivo | Conteúdo |
|---|---|
| `docs/AUDITORIA_TCC2_TCC3.md` | O que o TCC 2 pediu × o que foi entregue, lacuna a lacuna |
| `docs/RESULTADOS.md` | Todos os resultados experimentais e sua interpretação |
| `docs/DECISOES_METODOLOGICAS.md` | Divergências em relação ao projeto, com justificativa |
| `docs/T04_AMBIENTE_WSL2.md` | Os onze obstáculos vencidos para rodar os extratores de T04 |
| `docs/PLANO_ESCALA_INTEGRAL.md` | Viabilidade da escala e contingências |
| `external/CONTRATO.md` | Integração com os repositórios oficiais |

Repositório: <https://github.com/elianferreira/tcc3-deteccao-imagens-ia> (privado)

---

## PONTO DE RETOMADA

**Nada está em execução, e nenhuma lacuna de execução permanece.** Todas as
medições previstas no TCC 2 foram feitas e documentadas.

O que resta é **escrita**: redigir o Capítulo 4 a partir de `RESULTADOS.md`, que
traz seção a seção os números e as ressalvas de como reportar cada um.

### A pergunta em aberto, se houver tempo

A seção 4.13 de `RESULTADOS.md` diagnostica por que T04 fica em acaso: este
corpus oferece pouca estrutura geométrica. As sintéticas têm 49 segmentos de reta
por imagem contra 116 do corpus dos autores, e 46,5% não produzem par
objeto-sombra algum.

O diagnóstico **não separa** duas possibilidades: se o corpus de Corvi et al. é
geometricamente pobre como um todo, ou se apenas o `latent_diffusion` de 256 px
do split padrão o é. O benchmark tem SDXL, Midjourney, DALL·E 3 e Firefly, de
1024 px ou mais.

Resolver isso muda a conclusão do Capítulo 4 de *"o método não transfere"* para
*"o método exige geometria, e este gerador não a tem"*. Cerca de 2 h de extração;
os extratores e os scripts já existem — ver o fim da seção 4.13.

## Onde o trabalho chegou

### As quatro técnicas e a quinta

| Técnica | Protocolo padrão | Situação |
|---|---|---|
| T01 — Coocorrência + CNN | AUC 0,9962 | replicada, multi-semente 0,9964 ± 0,0002 |
| T02 — SPAI | AUC 0,9976 | replicada e **verificada** contra o publicado |
| T03 — Benford/DCT | AUC 0,8064 | replicada |
| T04 — Geometria projetiva | AUC 0,5333 | **três representações extraídas**; resultado negativo |
| **T05 — Fusão (contribuição)** | **AUC 0,9996** | medida com as quatro fontes completas |

### O que T04 estabeleceu

Os três extratores foram postos de pé no WSL2 — SSISv2, PerspectiveFields e
DeepLSD —, o que exigiu vencer **onze obstáculos** não documentados
(`T04_AMBIENTE_WSL2.md`). As três representações medem em acaso neste corpus:

| Representação | AUC |
|---|---|
| Objeto-sombra | 0,5384 |
| Campos de perspectiva | 0,5355 |
| Segmentos de reta | 0,5187 |
| Média das três | 0,5333 |

Os mesmos classificadores, sobre os dados dos autores, dão 0,82. A replicação
está correta; o que não transfere é o método. E a média **não** supera a melhor
representação — elas erram junto, não de formas complementares.

Na fusão, T04 é indiferente: com e sem ela T05 fica em 0,9996, e o teste de
DeLong dá **p = 0,182** com os dois modelos ajustados na mesma validação. Isso
confirma por medição a tolerância a fonte inútil prevista em RN07 e RNF04.

Detalhes em `RESULTADOS.md`, seções 4.7 a 4.12.

---

## Concluído

- Repositório publicado no GitHub
- Análise de erros e Grad-CAM de T01 (Etapa 5)
- Corpus em escala montado e regenerado sem perfil ICC: 192.638 imagens
- Os três manifestos da escala (padrão, OOD, OOD por famílias)
- Quatro confundidores investigados: resolução, formato, perfil ICC e
  **densidade de linhas**
- Multi-semente de T01 (Etapa 4) — AUC 0,9964 ± 0,0002
- **RNF01 completo** — soma 17,871 s contra o limite de 30 s
- **T02 verificada contra o valor publicado** (Etapa 3)
- **Os três extratores de T04**, e as três representações medidas
- **T05 medida com as quatro fontes completas**

---

## Decisões tomadas

### A campanha em escala não será executada

Decidido em 16/08/2026. Levaria de 2,5 a 4 dias e repetiria as mesmas medições
com mais dados. O corpus e os manifestos existem — a escala foi **viabilizada**,
que era o objetivo da Etapa 1 —, e a Seção 3 de `DECISOES_METODOLOGICAS.md` já
mostra que o colapso OOD persiste com 60.000 imagens de treino, isto é, não é
efeito de escala insuficiente.

**Reversível:** `scripts/pipeline_campanha.py --corpus data/corvi2024_escala
--prefixo escala`.

### T04 não entra na interface

Decidido em 17/08/2026. Com os extratores de pé, T04 passou a ser tecnicamente
executável sobre uma imagem enviada pela tela — mas os três modelos somam 1,5 GB
e vivem em outro sistema operacional, de modo que uma chamada por imagem gastaria
60 a 90 s só carregando, acima do limite de RNF01. Exigiria um serviço
persistente no WSL2.

Não construído: o comportamento atual já cumpre RN07, e T04 mede cerca de 0,53
neste corpus, de modo que a caixa na tela exibiria ruído. Justificativa completa
na seção 5 de `DECISOES_METODOLOGICAS.md`.

---

## Armadilhas conhecidas

Registradas porque cada uma já custou tempo.

1. **Compile antes de rodada longa.** Um `SyntaxError` custou 12 h de janela.
   `python -m compileall scripts src app tests`.
2. **Processos longos morrem sem traceback**, com log de erro vazio. Ocorreu
   cinco vezes. O Agendador protege contra o fechamento do shell, mas as tarefas
   estão em "Interativo apenas"; em 15/08 duas pararam às 23h55 sem causa
   determinada. Verificar o que está vivo antes de supor progresso.
3. **`schtasks /end` não derruba os filhos.** A GPU seguiu com 5,5 GB ocupados
   depois de encerrar a tarefa; só `taskkill /PID <pid> /T /F` liberou.
4. **Não sobrescrever os pesos do protocolo padrão.** Variantes devem usar
   `TCC3_WEIGHTS_DIR`, como fazem `pipeline_pos_multiseed.py` e
   `pipeline_t04_fusao.py`.
5. **O cache de escores não sabe que os dados de entrada mudaram.** Ao trocar os
   escores de T04 de um componente para os três, a campanha reaproveitou o
   `.npy` antigo — a forma batia — e reportou o valor errado em silêncio. Ao
   mudar a origem de um escore, apagar
   `results/scores/<protocolo>__<tecnica>__*`.
6. **Comparar modelos ajustados em ocasiões distintas mede o reajuste**, não a
   mudança que se quer testar. Um confronto assim sugeriu ganho significativo de
   T04 na fusão (p < 0,0001) que sumiu ao ajustar os dois na mesma validação
   (p = 0,182).
7. **O cache de escores colide entre manifestos.** A chave é
   `{protocolo}__{técnica}__{condição}__{split}`, e o protocolo é `ood` tanto
   para `manifesto30k_ood` quanto para `manifesto30k_ood_familias`. Hoje não
   causa erro porque as partições têm tamanhos diferentes e a guarda de forma em
   `runner.py:119` rejeita o incompatível — mas uma execução sobrescreve o cache
   da outra.
8. **T02 exige GPU livre.** As extrações de T04 no WSL2 também ocupam a placa e
   não convivem na RTX 3060 de 6 GB. Retas e campos de perspectiva rodam em CPU
   com `CUDA_VISIBLE_DEVICES` vazio, e aí sim convivem com treino na GPU.
9. **`wmic` não existe** nesta build do Windows 11.
10. **A interface não é afetada** pelos treinos: `run_multi_seed` grava em
    `weights/t01_seed<N>.pt`, sem tocar em `t01_cooccurrence.pt`.
