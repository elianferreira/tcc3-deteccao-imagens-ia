# Como rodar o sistema

Atualizado em 02/09/2026. Vale para os dois caminhos: **VS Code** e **terminal**.
Os tempos são medidos nesta máquina, não estimados.

---

## A regra que vem antes de tudo

> ## Serviços T04 primeiro. Interface depois.

A `is_available()` da T04 roda **na carga do Gradio**, uma vez só. Se a interface
subir antes dos serviços, a T04 fica indisponível pela sessão inteira — e aí
acontece o que é fácil não perceber:

**a interface troca de modelo de fusão sozinha.**

| T04 no ar? | Modelo carregado | Maior peso |
|---|---|---|
| sim | `t05_fusion__quatro_fontes.pkl` | T02 |
| não | `t05_fusion.pkl` (variante B) | T01 |

Os dois foram calibrados em partições — e protocolos — diferentes. A mesma
imagem (`adobe_firefly_00002`) já deu **2,5%** com um e **67,7%** com o outro.
Subir na ordem errada e não saber disso é ler um número achando que é outro.

**Como saber se deu certo:** a caixa da T04 aparece na tela com escore. Se disser
indisponível, pare tudo e suba na ordem.

---

## Caminho A — VS Code

### Passo 1. Abrir o projeto

Abra o VS Code e vá em **Arquivo → Abrir Pasta** →
`C:\Users\ferre\projects\tcc3-deteccao-imagens-ia`.

Se perguntar **"Do you trust the authors of the files in this folder?"**, clique
em **Yes, I trust the authors**. Sem isso o depurador não roda.

### Passo 2. Conferir o interpretador (só na primeira vez)

`Ctrl+Shift+P` → `Python: Select Interpreter` → escolha o que termina em
**`.venv\Scripts\python.exe`**.

Se escolher o Python do sistema, tudo quebra com `ModuleNotFoundError: torch`.

### Passo 3. Subir os serviços T04

`Ctrl+Shift+P` → `Tasks: Run Task` → **T04: subir serviços (WSL2)**.

Espere a mensagem verde. Se já estiverem no ar, a tarefa diz *"já estão no ar,
nada a fazer"* e sai — está certo, pode seguir.

### Passo 4. Rodar a interface

1. `Ctrl+Shift+D` (abre **Executar e Depurar**)
2. No menu suspenso do topo, escolha **Interface (Gradio) — configuração completa**
3. `F5`, ou o ▶ verde ao lado do menu

O terminal integrado abre embaixo. **A carga leva ~2 minutos** (medido: 112 s).
Quando aparecer a linha do Gradio, abra <http://localhost:7860>.

### Para parar

O quadrado vermelho ⏹ na barrinha de depuração, no topo. Ou `Shift+F5`.

### Outras coisas pelo VS Code

| O quê | Como |
|---|---|
| Todos os testes | `Ctrl+Shift+D` → **Testes (pytest, suíte completa)** → `F5` |
| Testes um a um | ícone do frasco 🧪 na barra lateral |
| Um script qualquer | abra o arquivo → **Arquivo Python aberto** → `F5` |
| Ver o que está vivo | `Tasks: Run Task` → **Estado: o que está vivo** |
| Compilar antes de rodada longa | `Tasks: Run Task` → **Compilar tudo (armadilha 1)** |

---

## Caminho B — terminal

É o que já rodava antes, e continua valendo. Use quando a rodada for longa: o
`F5` do VS Code amarra o processo à janela, e fechar a janela mata a rodada.

### Subir os serviços T04

```powershell
cd C:\Users\ferre\projects\tcc3-deteccao-imagens-ia
powershell -NoProfile -ExecutionPolicy Bypass -File .\.vscode\subir_servicos_t04.ps1
```

O script é o mesmo que a tarefa do VS Code chama. Ele detecta se já estão no ar
e não sobe em duplicidade.

### Subir a interface

```powershell
cd C:\Users\ferre\projects\tcc3-deteccao-imagens-ia
$env:TCC3_T04_SERVICO = "1"
$env:TCC3_T02_NATIVA  = "1"
$env:TCC3_T02_TETO    = "1536"
Start-Process -FilePath ".\.venv\Scripts\python.exe" `
  -ArgumentList "-u","app\gradio_app.py" `
  -WorkingDirectory (Get-Location).Path -WindowStyle Hidden
```

**Use `Start-Process`, não `python interface/gradio_app.py &`.** Processo lançado em
segundo plano de uma sessão morre junto com ela — é a armadilha nº 12.

### Parar a interface

```powershell
$p = Get-NetTCPConnection -State Listen | Where-Object LocalPort -eq 7860
taskkill /PID $p.OwningProcess /T /F
```

**O `/T` não é opcional:** sem ele os filhos sobrevivem e a placa continua
ocupada — é a armadilha nº 3.

### Conferir o que está vivo

```powershell
Get-Process python,pythonw | Select-Object Id,StartTime
Get-NetTCPConnection -State Listen | Where-Object LocalPort -in 7860,8404,8405
wsl -d Ubuntu-24.04 -u root -- bash -c "ps -eo etime,cmd | grep -E 'extrair_|servico_t04' | grep -v grep"
```

> **"Em execução" significa vivo. "Pronto" significa terminado *ou morto*** — e a
> diferença só aparece no log e nos arquivos de saída. Processos longos deste
> projeto já morreram cinco vezes sem deixar *traceback*.

---

## As variáveis de ambiente

O projeto lê nove. Três mudam o comportamento da tela e já estão fixadas nas
configurações do VS Code:

| Variável | Valor | O que muda |
|---|---|---|
| `TCC3_T04_SERVICO` | `1` | sem ela a T04 se declara indisponível **e o modelo de fusão troca em silêncio** |
| `TCC3_T02_NATIVA` | `1` | T02 em resolução nativa na tela (opção (b) de 31/08) |
| `TCC3_T02_TETO` | `1536` | teto que impede a T02 de esgotar a VRAM |

O teto não é cosmético: **sem ele a T02 nativa já derrubou os serviços T04 duas
vezes**, por esgotamento de VRAM, e a queda é silenciosa — as portas 8404/8405
simplesmente param de responder. Com teto o pico de GPU foi 3.108 MiB; sem,
5.931 MiB de 6.144.

As outras seis (`TCC3_WEIGHTS_DIR`, `TCC3_T04_ESCORES`, `TCC3_DATA_DIR`,
`TCC3_RESULTS_DIR`, `TCC3_EXTERNAL_DIR`, `TCC3_T04_SERVICO_TIMEOUT`) só entram
em campanhas específicas, e os scripts que precisam delas as definem sozinhos.

---

## Antes da defesa

> **Reinicie a interface pouco antes de demonstrar.**

Não é superstição, é número medido:

| Estado do processo | Primeira análise, 256 px |
|---|---|
| Recém-iniciado | **22,1 s** ✅ |
| 14 h ocioso | **54,2 s** ❌ |

O teto do RNF01 é 30 s. O `_aquecer()` da carga resolve o primeiro usuário
depois que a interface sobe; **não** resolve o processo que ficou horas parado
disputando 6 GB de VRAM com Teams, WhatsApp, Edge WebView e Kaspersky. Sem
reiniciar, o requisito documentado como cumprido é violado na frente da banca.

E lembre que a interface hoje leva ~2 min para carregar. Não deixe para os
últimos cinco minutos.

---

## Problemas comuns

| Sintoma | Causa provável | O que fazer |
|---|---|---|
| A tela diz que a T04 está indisponível | interface subiu antes dos serviços | pare a interface, suba os serviços, suba de novo |
| Os escores mudaram sem você mexer em nada | ordem errada trocou o modelo de fusão | mesma coisa: pare tudo, suba na ordem |
| Gradio subiu na porta 7861 | a 7860 já estava ocupada por outra instância | mate a antiga (ver "Parar a interface") e suba de novo |
| Primeira análise leva ~54 s | processo ocioso há horas; contexto CUDA despejado | reinicie a interface |
| `ModuleNotFoundError: torch` | interpretador errado no VS Code | `Ctrl+Shift+P` → **Python: Select Interpreter** → o do `.venv` |
| `code` não é reconhecido no PowerShell | o PATH só atualiza em terminal novo | feche e abra o PowerShell |
| VS Code trava ao abrir a pasta | indexação varrendo as 192.638 imagens de `data/` | confirme que `.vscode/settings.json` está no lugar |
| Extensão falha com `EPERM` ao instalar | antivírus segurando a pasta recém-extraída | rode o `--install-extension` de novo |
| Os serviços T04 pararam de responder | T02 em resolução nativa **sem teto** esgotou a VRAM | suba os serviços de novo e confirme `TCC3_T02_TETO=1536` |

---

## O ambiente do VS Code

Instalado em 02/09/2026: **VS Code 1.135.0**, instalação de usuário em
`%LOCALAPPDATA%\Programs\Microsoft VS Code`.

| Extensão | Para quê |
|---|---|
| `ms-python.python` | interpretador, execução, painel de testes |
| `ms-python.debugpy` | depurador |
| `ms-python.vscode-pylance` | autocompletar e checagem de tipos |
| `ms-vscode-remote.remote-wsl` | necessária para mexer nos extratores de T04 |

Para refazer noutra máquina:

```powershell
winget install --id Microsoft.VisualStudioCode --scope user --silent
# reabra o terminal para o comando `code` entrar no PATH
code --install-extension ms-python.python
code --install-extension ms-python.debugpy
code --install-extension ms-python.vscode-pylance
code --install-extension ms-vscode-remote.remote-wsl
```

### Os arquivos em `.vscode/`

**São versionados de propósito.** Não é preferência pessoal: o `launch.json`
carrega variáveis sem as quais os números mudam, e o `tasks.json` guarda a ordem
de subida. Isso é conhecimento do projeto.

| Arquivo | O que faz |
|---|---|
| `settings.json` | interpretador do `.venv`, pytest ligado, exclusão de `data/`, `pesos/`, `externo/` da indexação |
| `launch.json` | nove configurações de execução, com as variáveis certas em cada uma |
| `tasks.json` | subir/parar/conferir T04, compilar, testar, ver o que está vivo |
| `subir_servicos_t04.ps1` | o script que a tarefa de subida chama |
| `extensions.json` | as quatro extensões, para o VS Code oferecer sozinho |

O corpus tem 192.638 imagens e `externo/` traz os repositórios de terceiros.
Sem as exclusões, o VS Code varre tudo ao abrir e a janela trava por minutos.

---

## O que o VS Code **não** resolve

Três limites reais. Nenhum é defeito da configuração.

### 1. Os extratores de T04 não rodam no Windows

Vivem no WSL2, com interpretador próprio (`/root/geo/bin/python`), sobre SSIS,
PerspectiveFields e DeepLSD. Nada disso existe do lado Windows.

Para editar ou depurar **os extratores**, abra uma segunda janela:
`Ctrl+Shift+P` → **WSL: Connect to WSL using Distro** → `Ubuntu-24.04`, e abra
`/mnt/c/Users/ferre/projects/tcc3-deteccao-imagens-ia`. É outra raiz e outro
interpretador — selecione `/root/geo/bin/python`. A configuração deste
repositório não vale lá, porque aponta para o `.venv` do Windows.

### 2. A T02 roda em venv separado, por subprocesso

`codigo/tecnicas/t02_spai/tecnica.py:173` lança o Python do repositório clonado do SPAI
como **processo filho**. Breakpoint dentro do código do SPAI **não dispara** — o
depurador está preso ao processo principal.

O que funciona é breakpoint na fronteira: antes da chamada, para ver o que é
enviado; depois, para ver o que volta. Para a maior parte do trabalho basta,
porque a T02 não é retreinável e o interesse está nas entradas e saídas.

### 3. Rodadas longas continuam sendo do terminal

Extração de T04, campanhas, benchmark — horas de execução. O `.bat` com
`Start-Process` é o que já sobreviveu a essas janelas. Pelo `F5` funciona, mas
fechar a janela do VS Code mata a rodada.

### 4. Não dá para iniciar a depuração de fora

Tentado em 02/09/2026 e **não funciona**: automação de teclado (`SendKeys`) para
a janela do VS Code não dispara o `F5`. Iniciar uma sessão de depuração exige a
tecla na janela, com alguém na frente. Se precisar subir a interface sem
interação — de script, de tarefa agendada —, use o **Caminho B**.

---

## Onde continuar lendo

| Arquivo | Conteúdo |
|---|---|
| `documentacao/ESTADO_ATUAL.md` | o documento de retomada; as armadilhas conhecidas estão no fim |
| `documentacao/RESULTADOS.md` | os números e as ressalvas de como reportar cada um |
| `documentacao/T04_AMBIENTE_WSL2.md` | os onze obstáculos vencidos para pôr os extratores de pé |
