@echo off
REM ===================================================================
REM T04 sobre os geradores modernos do benchmark.
REM
REM Responde a pergunta em aberto do docs/ESTADO_ATUAL.md: T04 fica em
REM acaso por causa do corpus inteiro, ou so por causa do latent_diffusion
REM de 256 px do split padrao?
REM
REM Alvo: split "test" de data/manifesto30k_ood.csv -- 13.788 imagens,
REM 11 geradores modernos (SDXL, Midjourney v5/v6.1, DALL-E 2/3, Firefly,
REM Flux, GigaGAN, SD 1.3/1.4/2/3) mais 3.150 reais.
REM
REM -------------------------------------------------------------------
REM POR QUE A ETIQUETA DE CONJUNTO E "tcc3_ood" E NAO "tcc3_30k"
REM -------------------------------------------------------------------
REM Trocar apenas o manifesto e manter a etiqueta faria esta rodada
REM SOBRESCREVER os mapas e escores que produziram os numeros da
REM dissertacao (0,5384 / 0,5355 / 0,5187). E a armadilha 5 do
REM ESTADO_ATUAL.md, que ja custou um resultado errado reportado em
REM silencio. A etiqueta separada mantem as duas rodadas lado a lado.
REM
REM -------------------------------------------------------------------
REM POR QUE OS DISPOSITIVOS SAO ESTES
REM -------------------------------------------------------------------
REM Sao os mesmos da rodada padrao -- objeto-sombra em GPU, campos e
REM retas em CPU. Nao e detalhe: medido em 30/08/2026, trocar CPU por
REM GPU nos campos altera o escore em ate 2,2e-02. Com dispositivos
REM diferentes, os numeros desta rodada nao seriam comparaveis aos da
REM dissertacao, e a comparacao e justamente o objetivo.
REM
REM As retas EXIGEM CUDA_VISIBLE_DEVICES vazio: o PointNet dos autores
REM manda a matriz identidade para CUDA sempre que a placa estiver
REM visivel (lines_model.py:39-41).
REM
REM -------------------------------------------------------------------
REM TEMPO
REM -------------------------------------------------------------------
REM Estimativa: 5 a 6 h em sequencia. Objeto-sombra ~45 min na GPU;
REM campos e retas somam ~1,3 s por imagem em CPU. As duas etapas de CPU
REM podem rodar em paralelo com a de GPU -- ver a nota no final.
REM
REM Toda etapa e retomavel: itens ja gravados sao pulados.
REM
REM Uso:
REM   scripts\t04_benchmark.bat              -> as tres etapas
REM   scripts\t04_benchmark.bat sombra       -> so objeto-sombra
REM   scripts\t04_benchmark.bat campos       -> so campos
REM   scripts\t04_benchmark.bat retas        -> so retas
REM ===================================================================

@echo on
cd /d "%~dp0.."

set RAIZ=/mnt/c/Users/ferre/projects/tcc3-deteccao-imagens-ia
set MANIFESTO=%RAIZ%/data/manifesto30k_ood.csv
set CONJUNTO=tcc3_ood
set ETAPA=%1
if "%ETAPA%"=="" set ETAPA=todas

if not exist logs mkdir logs

if "%ETAPA%"=="todas" goto :sombra
if "%ETAPA%"=="sombra" goto :sombra
if "%ETAPA%"=="campos" goto :campos
if "%ETAPA%"=="retas" goto :retas
echo Etapa desconhecida: %ETAPA%
exit /b 1

:sombra
echo === objeto-sombra (GPU) ===
wsl -d Ubuntu-24.04 -u root -- bash -c "cd /root/SSIS/demo && /root/geo/bin/python %RAIZ%/scripts/wsl/extrair_object_shadow_ssis.py --manifesto %MANIFESTO% --split test --saida %RAIZ%/data/t04_object_shadow --conjunto %CONJUNTO%" >> logs\t04_bench_sombra.log 2>> logs\t04_bench_sombra.err
echo >>> objeto-sombra: codigo %ERRORLEVEL%
if not "%ETAPA%"=="todas" exit /b %ERRORLEVEL%

:campos
echo === campos de perspectiva (CPU) ===
wsl -d Ubuntu-24.04 -u root -- bash -c "cd /root && CUDA_VISIBLE_DEVICES='' /root/geo/bin/python %RAIZ%/scripts/wsl/extrair_perspective_fields.py --manifesto %MANIFESTO% --split test --saida %RAIZ%/data/t04_perspective_fields --conjunto %CONJUNTO% --dispositivo cpu" >> logs\t04_bench_campos.log 2>> logs\t04_bench_campos.err
echo >>> campos: codigo %ERRORLEVEL%
if not "%ETAPA%"=="todas" exit /b %ERRORLEVEL%

:retas
echo === segmentos de reta (CPU) ===
wsl -d Ubuntu-24.04 -u root -- bash -c "cd /root/DeepLSD && CUDA_VISIBLE_DEVICES='' /root/geo/bin/python %RAIZ%/scripts/wsl/extrair_line_segments.py --manifesto %MANIFESTO% --split test --saida %RAIZ%/data/t04_line_segments --conjunto %CONJUNTO% --dispositivo cpu" >> logs\t04_bench_retas.log 2>> logs\t04_bench_retas.err
echo >>> retas: codigo %ERRORLEVEL%

echo.
echo ===================================================================
echo Extracao concluida. Proximos passos, NESTA ordem:
echo.
echo   REM 1. classifica os mapas objeto-sombra (o --sufixo evita
echo   REM    sobrescrever results/t04_escores_combined.csv, da dissertacao)
echo   python scripts\avaliar_t04_corpus.py --conjunto tcc3_ood --splits test --sufixo _ood
echo.
echo   REM 2. junta as tres representacoes desta rodada, e so dela
echo   python scripts\consolidar_escores_t04.py --conjunto tcc3_ood --sufixo _ood ^
echo          --saida results\t04_escores_componentes_ood.csv
echo.
echo   REM 3. AUC por representacao
echo   python scripts\avaliar_t04_componentes.py --split test ^
echo          --escores results\t04_escores_componentes_ood.csv
echo.
echo Depois, comparar a AUC POR GERADOR. E isso que responde a pergunta:
echo se T04 sobe nos geradores de 1024 px e cai no latent_diffusion, o
echo metodo exige geometria; se fica em 0,53 em todos, nao transfere.
echo ===================================================================
exit /b 0
