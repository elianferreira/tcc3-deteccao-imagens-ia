# Estado atual e como retomar

Atualizado em 19/08/2026.

Este arquivo existe para que o trabalho possa ser retomado em outra sessão sem
depender do histórico da conversa. Tudo que importa está no repositório.

---

## ❓ PERGUNTA EM ABERTO — retomar por aqui

> **T04 falha por causa do corpus inteiro, ou só por causa do
> `latent_diffusion` de 256 px?**

As três representações de T04 medem em acaso (0,52 a 0,54) sobre o split padrão.
A seção 4.13 de [`RESULTADOS.md`](RESULTADOS.md) mostra por quê: este corpus
oferece pouca estrutura geométrica — as sintéticas têm **49 segmentos de reta**
por imagem contra **116** do corpus dos autores, e 46,5% não produzem par
objeto-sombra algum.

Mas o split padrão tem **um único gerador sintético**, o `latent_diffusion` de
256 px nativos, que é antigo e produz imagens suaves. O benchmark tem SDXL,
Midjourney v5/v6.1, DALL·E 3 e Firefly — de 1024 px ou mais, com cenas
estruturadas.

**Por que importa.** A conclusão do Capítulo 4 muda de *"o método geométrico não
transfere"* para *"o método exige imagens com geometria, e este gerador não a
tem"*. São afirmações bem diferentes para a banca.

**Como responder.** Extrair as três representações sobre o split `test` de
`data/manifesto30k_ood.csv` — **13.788 imagens**, 11 geradores modernos (SDXL,
Midjourney v5/v6.1, DALL·E 2/3, Firefly, Flux, GigaGAN, SD 1.3/1.4/2/3) mais
3.150 reais — e comparar a AUC **por gerador**.

Um comando faz as três etapas, com os dispositivos já casados aos da rodada
padrão e a etiqueta de conjunto separada:

```powershell
scripts	04_benchmark.bat
```

Depois, nesta ordem:

```powershell
python scriptsvaliar_t04_corpus.py --conjunto tcc3_ood --splits test --sufixo _ood
python scripts\consolidar_escores_t04.py --conjunto tcc3_ood --sufixo _ood ^
       --saida results	04_escores_componentes_ood.csv
python scriptsvaliar_t04_componentes.py --split test ^
       --escores results	04_escores_componentes_ood.csv
```

**Tempo: 5 a 6 h em sequência**, não as 2 h estimadas antes — a estimativa
anterior usava 12.638 imagens (contagem errada) e supunha GPU para os campos.
As etapas de CPU podem rodar em paralelo com a de GPU, o que traz para ~4 h.

**Três armadilhas, todas já neutralizadas no código em 30/08/2026** — registradas
porque explicam por que os comandos têm esses argumentos:

1. `--conjunto tcc3_ood` impede que os mapas desta rodada sobrescrevam os da
   dissertação (armadilha 5).
2. `--sufixo _ood` impede que `avaliar_t04_corpus.py` grave por cima de
   `results/t04_escores_combined.csv`, que é a fonte da AUC 0,5384. O nome do
   arquivo de saída não continha a etiqueta de conjunto.
3. `--conjunto` em `consolidar_escores_t04.py` impede que ele varra o diretório
   inteiro e **concatene as duas rodadas em silêncio**, misturando dois corpora.
   Era o comportamento anterior.

Os dispositivos no `.bat` (objeto-sombra em GPU, campos e retas em CPU) são os
mesmos da rodada padrão, e isso não é detalhe: trocar CPU por GPU nos campos
altera o escore em até 2,2e-02, e a comparação com os números da dissertação é
justamente o objetivo.

**Como ler o resultado.** Se a T04 subir nos geradores de 1024 px e ficar em
acaso no `latent_diffusion`, o método exige geometria e o corpus é que não a
tem. Se ficar em 0,53 em todos, o método não transfere. Os dois desfechos são
reportáveis.

**É opcional.** Todas as medições exigidas pelo TCC 2 já estão feitas; esta
apenas qualificaria melhor um resultado negativo.

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

## Trabalho futuro planejado

Registrado em 30/08/2026, a pedido. **Nada aqui foi executado.**

### 1. Campanha em escala integral

O treino atual usa **42.000 imagens** (split `train` de
`data/manifesto30k_standard.csv`, corpus de 60.000). A escala integral leva o
treino a **126.000** (`data/manifesto_escala_standard.csv`, corpus de 180.000
sobre as 192.638 montadas em disco).

Isto reverte a decisão de 16/08 de não executar a escala. Comando registrado na
seção "Decisões tomadas" deste arquivo.

### 2. Geradores de 2025/2026 — testar antes de treinar

Duas etapas, **nesta ordem**, e a ordem é o ponto:

1. **Testar primeiro.** Avaliar a arquitetura já treinada sobre imagens de
   geradores atuais, sem retreinar nada. Isso mede se ela **se mantém boa** em
   material que não existia quando o corpus foi montado — é uma medição de
   generalização temporal, e o resultado vale por si, positivo ou negativo.
2. **Treinar depois.** Só então incorporar esses geradores ao treino.

Inverter a ordem destruiria a medição: uma vez que os geradores novos entram no
treino, não há mais como saber como a arquitetura se comportava sem eles. É o
mesmo raciocínio que torna o protocolo OOD informativo.

O corpus de teste atual já cobre até Flux e Midjourney v6.1; o que falta são
geradores posteriores à montagem do corpus.

---

## PONTO DE RETOMADA

**Nada está em execução, e nenhuma lacuna de execução permanece.** Todas as
medições previstas no TCC 2 foram feitas e documentadas.

O que resta é **escrita**: redigir o Capítulo 4 a partir de `RESULTADOS.md`, que
traz seção a seção os números e as ressalvas de como reportar cada um.

Ver a **pergunta em aberto** no topo deste arquivo.

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

### T04 entrou na interface — decisão de 17/08 revertida em 30/08

Decidido em 17/08/2026 manter T04 fora da tela, por estimativa de 1,5 GB de
modelos e 60 a 90 s de carga por imagem, acima do teto de 30 s do RNF01.

**A estimativa nunca fora medida, e estava errada:** 883 MiB e ~10 s. Com um
serviço residente no WSL2, T04 passou a pontuar imagens arbitrárias.

| | Estimado | Medido |
|---|---|---|
| VRAM | ~1,5 GB | 883 MiB |
| Carga | 60–90 s | ~10 s |
| RNF01 com T04 | violaria | **20,03 s** (teto 30 s) |

Construído:

| Arquivo | O quê |
|---|---|
| `scripts/wsl/servico_t04.py` | serviço residente, dois processos |
| `scripts/wsl/servico_t04_controle.sh` | iniciar/parar/consultar no WSL |
| `scripts/verificar_paridade_servico_t04.py` | valida contra os escores da dissertação |
| `scripts/medir_rnf01_com_t04.py` | remede o RNF01 pelo caminho da interface |

**Como subir** (as duas portas, com os dispositivos casados ao lote):

```powershell
# 8404: objeto-sombra em GPU
Start-Process wsl.exe -ArgumentList "-d","Ubuntu-24.04","-u","root","--",
  "/root/geo/bin/python","-u","/mnt/c/.../scripts/wsl/servico_t04.py",
  "--porta","8404","--dispositivo","cuda","--representacoes","object_shadow"

# 8405: campos e retas em CPU, com a placa escondida
Start-Process wsl.exe -ArgumentList "-d","Ubuntu-24.04","-u","root","--","env",
  "CUDA_VISIBLE_DEVICES=","/root/geo/bin/python","-u","/mnt/c/.../servico_t04.py",
  "--porta","8405","--dispositivo","cpu",
  "--representacoes","perspective_fields,line_segment"
```

Depois, a interface com `TCC3_T04_SERVICO=1`. **Sem essa variável nada muda** —
T04 volta a se declarar indisponível, como antes.

Paridade: campos e retas exatos até 1e-16; objeto-sombra em 2,3e-04, resíduo de
cuDNN com lote 1 contra lote 128. Justificativa completa na seção 5 de
`DECISOES_METODOLOGICAS.md`.

**O resultado não muda:** T04 mede 0,53 e a caixa exibe ruído. O ganho é de
demonstrabilidade da arquitetura de quatro fontes.

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
11. **Um processo não serve as três representações de T04.** O PointNet dos
    autores (`lines_model.py:39-41`) manda a matriz identidade para CUDA sempre
    que `torch.cuda.is_available()` for verdadeiro — não quando o modelo está na
    GPU. O contorno é `CUDA_VISIBLE_DEVICES=""`, que vale para o processo
    inteiro e conflita com o objeto-sombra, que roda em GPU. Daí dois serviços.
12. **Processo lançado de dentro do `wsl.exe` não sobrevive ao retorno da
    chamada** — nem com `setsid`, nem com `nohup`. Morre sem escrever no log,
    parecendo a armadilha nº 2. O que segura é lançar pelo lado Windows com
    `Start-Process wsl.exe ...`, que mantém um processo vivo no Windows
    ancorando o do WSL. Vale igual para o Gradio: `python app/gradio_app.py`
    em segundo plano da sessão morre junto com ela.
