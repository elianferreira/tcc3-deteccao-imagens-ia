@echo off
REM ===================================================================
REM Envelope de agendamento para scripts\t04_benchmark.bat.
REM
REM Existe para que a rodada sobreviva ao fechamento da sessao: quem
REM chama isto e a tarefa \tcc3_t04_benchmark do Agendador do Windows,
REM nao um terminal. Nao muda nada do metodo -- so carimba inicio/fim e
REM manda toda a saida de console para um log.
REM ===================================================================
cd /d "%~dp0.."
if not exist logs mkdir logs
echo [%DATE% %TIME%] INICIO da rodada tcc3_ood (todas as etapas)>>logs\t04_bench_status.log
call scripts\t04_benchmark.bat >>logs\t04_bench_master.log 2>&1
set COD=%ERRORLEVEL%
echo [%DATE% %TIME%] FIM da rodada tcc3_ood: codigo %COD% >>logs\t04_bench_status.log
exit /b %COD%
