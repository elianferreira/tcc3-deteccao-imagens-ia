@echo off
REM Etapas dependentes de GPU, apos o termino do treino multi-semente:
REM RNF01 com T02 e o protocolo OOD por familias da Secao 3.6.1.
REM
REM Executado pelo Agendador de Tarefas pelo mesmo motivo do multiseed.bat:
REM processos lancados pelo shell da sessao foram encerrados sem traceback.

cd /d "%~dp0.."
.venv\Scripts\python.exe -u scripts\pipeline_pos_multiseed.py ^
  --timeout-horas 14 >> logs\pos_multiseed.log 2>> logs\pos_multiseed.err
