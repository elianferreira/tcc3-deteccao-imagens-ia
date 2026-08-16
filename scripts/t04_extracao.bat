@echo off
REM Extracao dos mapas objeto-sombra de T04, via SSISv2 no WSL2.
REM
REM Executado pelo Agendador de Tarefas pelo mesmo motivo do multiseed.bat:
REM processos lancados pelo shell da sessao foram encerrados sem traceback,
REM inclusive com DETACHED_PROCESS. A rodada completa sobre o manifesto padrao
REM leva cerca de 3,5 h (60.000 imagens a 0,21 s cada).
REM
REM Ocupa a GPU: nao deve rodar junto com o pipeline de T02.

REM Aceita os splits como argumentos; sem argumentos, faz os tres.
REM   schtasks /run ...                      -> test train val
REM   scripts\t04_extracao.bat train         -> so o treino

cd /d "%~dp0.."
wsl -d Ubuntu-24.04 -u root -- bash /mnt/c/Users/ferre/projects/tcc3-deteccao-imagens-ia/scripts/wsl/extrair_object_shadow.sh %* >> logs\t04_extracao.log 2>> logs\t04_extracao.err
