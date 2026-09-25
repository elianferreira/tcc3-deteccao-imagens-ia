@echo off
setlocal
rem ===================================================================
rem  A T05 de QUATRO fontes na troca de fonte real — a pergunta da A3.
rem
rem  POR QUE ESTA RODADA
rem  -------------------
rem  Em 24/09 mediu-se a T05 sobre o RAISE-1k e ela desabou (4,3% -> 47,4%).
rem  Mas o modelo carregado era `pesos/t05_fusion.pkl`, que e a variante B e
rem  tem coeficiente de T04 **exatamente zero** — ou seja, uma fusao de tres
rem  fontes. A de quatro existe em disco e nunca foi medida.
rem
rem    pesos/t05_fusion.pkl               [2,671 2,027 1,146  0,000]  3 fontes
rem    pesos_t04_fusao/t05_fusion.pkl     [3,430 4,266 0,521 -0,251]  4 fontes
rem    pesos_t05_ood_fusion/t05_fusion.pkl[2,662 2,025 1,144 +0,251]  4 fontes
rem
rem  ⚠️ Os dois de quatro fontes DISCORDAM NO SINAL da T04. Por isso as duas
rem  variantes rodam: qual delas se comporta como o texto descreve e uma das
rem  coisas que esta rodada responde.
rem
rem  T01, T03 e SPAI sao BYTE-IDENTICOS nos tres diretorios (conferido por md5
rem  em 25/09), entao a unica variavel entre as rodadas e o modelo da fusao.
rem
rem  NAO PRECISA DOS SERVICOS T04
rem  ----------------------------
rem  Os escores da T04 ja estao extraidos (`t04_escores_raise1k.csv` e
rem  `t04_escores_coco_controle.csv`, 999 cada, de 24/09) e sao lidos de CSV.
rem  Como a T02 roda aqui, e melhor que os servicos fiquem FORA: eles disputam
rem  a VRAM com ela na RTX 3060 de 6 GB (armadilha n 8).
rem
rem  CADA VARIANTE GRAVA NA PROPRIA PASTA (--sufixo), porque sobrescrever
rem  resultado ja custou os numeros da dissertacao uma vez.
rem ===================================================================

set "RAIZ=C:\Users\ferre\projects\tcc3-deteccao-imagens-ia"
set "PY=%RAIZ%\.venv\Scripts\python.exe"
set "LOG=%RAIZ%\logs\t05_quatro_fontes.log"

cd /d "%RAIZ%"
if not exist "%RAIZ%\logs" mkdir "%RAIZ%\logs"

>>"%LOG%" echo.
>>"%LOG%" echo ==================================================
>>"%LOG%" echo INICIO %DATE% %TIME%
>>"%LOG%" echo ==================================================

rem --- a tela e os servicos T04 ocupam a placa e nao sao necessarios ---
for /f "tokens=5" %%P in ('netstat -ano ^| findstr /r /c:":7860 .*LISTENING"') do (
  >>"%LOG%" echo derrubando a tela, pid %%P
  taskkill /PID %%P /T /F >>"%LOG%" 2>&1
)
wsl -d Ubuntu-24.04 -u root -- pkill -f servico_t04.py >>"%LOG%" 2>&1
>>"%LOG%" echo servicos T04 fora (pkill devolveu %ERRORLEVEL%) — a T02 precisa da VRAM

rem =============== VARIANTE 1: quatro fontes, T04 negativa ===============
>>"%LOG%" echo.
>>"%LOG%" echo --- pesos_t04_fusao (coef T04 = -0,251) ---
set "TCC3_WEIGHTS_DIR=%RAIZ%\pesos_t04_fusao"
"%PY%" -u "%RAIZ%\automacao\fpr_raise1k_todas.py" --com-t04 --sem-cofre --sufixo __t04_fusao >>"%LOG%" 2>&1
>>"%LOG%" echo codigo: %ERRORLEVEL%

rem =============== VARIANTE 2: quatro fontes, T04 positiva ===============
>>"%LOG%" echo.
>>"%LOG%" echo --- pesos_t05_ood_fusion (coef T04 = +0,251) ---
set "TCC3_WEIGHTS_DIR=%RAIZ%\pesos_t05_ood_fusion"
"%PY%" -u "%RAIZ%\automacao\fpr_raise1k_todas.py" --com-t04 --sem-cofre --sufixo __ood_fusion >>"%LOG%" 2>&1
>>"%LOG%" echo codigo: %ERRORLEVEL%

>>"%LOG%" echo.
>>"%LOG%" echo A saida detalhada de cada rodada esta em logs\fpr_raise1k_todas.log
>>"%LOG%" echo (o script espelha o proprio stdout para la e silencia o console
>>"%LOG%" echo  quando nao ha TTY — nao e perda de log).
>>"%LOG%" echo FIM %DATE% %TIME%
