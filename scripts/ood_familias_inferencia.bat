@echo off
REM Retoma o protocolo OOD por familias a partir do ponto em que parou.
REM
REM NAO usa --fit de proposito. O treino ja terminou e os modelos estao em
REM weights_ood_familias/ (T01, T03, T05); relancar o pipeline completo
REM retreinaria tudo, jogando fora ~8 h de trabalho.
REM
REM Sem --fit os modelos sao carregados e os escores ja calculados vem do cache
REM em results/scores/. Falta recalcular T02 e T03 sobre o split de teste.

cd /d "%~dp0.."
set TCC3_WEIGHTS_DIR=%~dp0..\weights_ood_familias

.venv\Scripts\python.exe -u scripts\run_experiments.py ^
  --protocol ood ^
  --manifest data\manifesto30k_ood_familias.csv ^
  --techniques T01 T02 T03 T05 ^
  --refit-fusion >> logs\ood_familias.log 2>> logs\ood_familias.err
