@echo off
REM Extracao dos campos de perspectiva de T04, via PerspectiveFields no WSL2.
REM
REM Aceita os splits e o dispositivo como argumentos:
REM   scripts\t04_campos.bat val cpu     -> convive com treino na GPU
REM   scripts\t04_campos.bat test cuda   -> com a GPU livre
REM
REM Em CPU custa ~1,4 s por imagem; em GPU, ~0,2 s. Rodar a validacao em CPU
REM enquanto a GPU esta ocupada por outra etapa usa os dois recursos em paralelo
REM em vez de esperar.

setlocal enabledelayedexpansion
cd /d "%~dp0.."

set SPLITS=%1
set DISPOSITIVO=%2
if "%SPLITS%"=="" set SPLITS=test
if "%DISPOSITIVO%"=="" set DISPOSITIVO=cuda

for %%S in (%SPLITS%) do (
  wsl -d Ubuntu-24.04 -u root -- bash -c "cd /root && /root/geo/bin/python /mnt/c/Users/ferre/projects/tcc3-deteccao-imagens-ia/scripts/wsl/extrair_perspective_fields.py --manifesto /mnt/c/Users/ferre/projects/tcc3-deteccao-imagens-ia/data/manifesto30k_standard.csv --split %%S --saida /mnt/c/Users/ferre/projects/tcc3-deteccao-imagens-ia/data/t04_perspective_fields --conjunto tcc3_30k --dispositivo %DISPOSITIVO%" >> logs\t04_campos.log 2>> logs\t04_campos.err
  echo ^>^>^> split %%S: codigo !ERRORLEVEL! >> logs\t04_campos.log
)
