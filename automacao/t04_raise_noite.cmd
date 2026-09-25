@echo off
setlocal
rem ===================================================================
rem  A T04 sobre o RAISE-1k e o controle do COCO, e a FPR de quatro fontes.
rem
rem  DUAS PASSAGENS, E A ORDEM E O PONTO
rem  ------------------------------------
rem  A T02 em resolucao nativa e os extratores da T04 nao cabem juntos nos
rem  6 GB da RTX 3060 (armadilha n 8; medido em 21/09, 10 CUDA out of memory
rem  em 26 envios). Entao:
rem
rem    passagem 1   servicos T04 no ar, T02 parada   -> extrai, grava CSV
rem    (derruba os servicos, libera a placa)
rem    passagem 2   servicos parados, T02 roda       -> le os CSVs, mede
rem
rem  Nenhuma das duas reexecuta o que a outra fez.
rem
rem  ARMADILHAS RESPEITADAS AQUI
rem  ---------------------------
rem   n 1  scripts compilados antes da rodada (feito em 23/09, na sessao)
rem   n 3  parar servico nao basta: /T para levar os filhos junto
rem   n 12 processo lancado de dentro do wsl.exe nao sobrevive -> o .ps1
rem        do lado Windows e quem sobe os servicos
rem   tarefa do Agendador nao tem console: tudo redirecionado, python com -u
rem   em .bat, "echo texto %VAR%>>arquivo" nao escreve -> redirecionar antes
rem ===================================================================

set "RAIZ=C:\Users\ferre\projects\tcc3-deteccao-imagens-ia"
set "PY=%RAIZ%\.venv\Scripts\python.exe"
set "LOG=%RAIZ%\logs\t04_raise_noite.log"

cd /d "%RAIZ%"
if not exist "%RAIZ%\logs" mkdir "%RAIZ%\logs"

>>"%LOG%" echo.
>>"%LOG%" echo ==================================================
>>"%LOG%" echo INICIO %DATE% %TIME%
>>"%LOG%" echo ==================================================

rem --- a tela ocupa a placa e nao e necessaria aqui ---
for /f "tokens=5" %%P in ('netstat -ano ^| findstr /r /c:":7860 .*LISTENING"') do (
  >>"%LOG%" echo derrubando a tela, pid %%P
  taskkill /PID %%P /T /F >>"%LOG%" 2>&1
)

rem =================== PASSAGEM 1 ====================
>>"%LOG%" echo.
>>"%LOG%" echo --- passagem 1: subindo os servicos T04 ---
powershell -NoProfile -ExecutionPolicy Bypass -File "%RAIZ%\.vscode\subir_servicos_t04.ps1" >>"%LOG%" 2>&1

>>"%LOG%" echo.
>>"%LOG%" echo --- extraindo T04 sobre o RAISE-1k (999) ---
"%PY%" -u "%RAIZ%\automacao\extrair_t04_raise1k.py" --conjunto raise1k >>"%LOG%" 2>&1
set "COD1=%ERRORLEVEL%"
>>"%LOG%" echo codigo da extracao raise1k: %COD1%

>>"%LOG%" echo.
>>"%LOG%" echo --- extraindo T04 sobre o controle do COCO (999) ---
"%PY%" -u "%RAIZ%\automacao\extrair_t04_raise1k.py" --conjunto coco >>"%LOG%" 2>&1
set "COD2=%ERRORLEVEL%"
>>"%LOG%" echo codigo da extracao coco: %COD2%

rem --- libera a placa antes da T02 ---
rem  Nao chamamos servico_t04_controle.sh: o arquivo esta com finais de linha
rem  CRLF e o bash do WSL recusa ("$'\r': command not found", erro de sintaxe
rem  no `case`). O pkill faz o servico e e o que de fato derrubou as duas
rem  portas no teste de 23/09.
>>"%LOG%" echo.
>>"%LOG%" echo --- derrubando os servicos T04 para liberar a VRAM ---
wsl -d Ubuntu-24.04 -u root -- pkill -f servico_t04.py >>"%LOG%" 2>&1
>>"%LOG%" echo servicos T04 encerrados (pkill devolveu %ERRORLEVEL%)

if not "%COD1%"=="0" goto :interrompe
if not "%COD2%"=="0" goto :interrompe

rem =================== PASSAGEM 2 ====================
>>"%LOG%" echo.
>>"%LOG%" echo --- passagem 2: FPR das cinco tecnicas, T05 de quatro fontes ---
"%PY%" -u "%RAIZ%\automacao\fpr_raise1k_todas.py" --com-t04 >>"%LOG%" 2>&1
set "COD3=%ERRORLEVEL%"
>>"%LOG%" echo codigo da medicao: %COD3%

>>"%LOG%" echo.
>>"%LOG%" echo FIM %DATE% %TIME%
exit /b %COD3%

:interrompe
>>"%LOG%" echo.
>>"%LOG%" echo ABORTADO: uma das extracoes falhou, a medicao nao roda sobre
>>"%LOG%" echo dado incompleto. Os CSVs parciais ficam em resultados\ e a
>>"%LOG%" echo proxima execucao retoma de onde parou.
>>"%LOG%" echo FIM %DATE% %TIME%
exit /b 1
