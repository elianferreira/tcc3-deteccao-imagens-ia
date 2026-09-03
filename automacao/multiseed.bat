@echo off
REM Treino multi-semente de T01, para execucao pelo Agendador de Tarefas.
REM
REM Processos lancados a partir do shell da sessao -- inclusive com
REM DETACHED_PROCESS -- foram encerrados duas vezes nesta maquina, sem deixar
REM traceback: os arquivos de erro ficaram com zero byte. O Agendador executa a
REM tarefa em contexto proprio, independente de qualquer shell.
REM
REM run_multi_seed retoma sementes ja concluidas, entao reexecutar e seguro.

cd /d "%~dp0.."
.venv\Scripts\python.exe -u automacao\run_experiments.py ^
  --protocol standard ^
  --manifest data\manifesto30k_standard.csv ^
  --multi-seed ^
  --techniques T01 >> logs\multiseed.log 2>> logs\multiseed.err
