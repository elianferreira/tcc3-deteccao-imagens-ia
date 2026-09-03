@echo off
REM ===================================================================
REM T04 sobre a particao "fusion" do protocolo OOD -- as 1.000 do glide.
REM
REM POR QUE ESTA RODADA EXISTE
REM -------------------------------------------------------------------
REM Destrava a recalibracao da T05. O modelo que a interface carrega
REM (t05_fusion__quatro_fontes.pkl) foi ajustado com --fusion-split val,
REM que e in-distribution -- o regime que o DECISOES_METODOLOGICAS.md
REM identifica como o que quebra a fusao. O reajuste correto e no
REM protocolo OOD com --fusion-split fusion, que e onde a variante B foi
REM ajustada.
REM
REM O bloqueio era falta de dado: a particao "fusion" do
REM data/manifesto30k_ood.csv tem 2.350 imagens -- 1.350 reais (coco) e
REM 1.000 do glide -- e os escores de T04 cobriam apenas as reais.
REM Cobertura de 57,4%, e ZERO nas sinteticas. Esta rodada extrai as
REM 1.000 que faltam, nas tres representacoes.
REM
REM -------------------------------------------------------------------
REM POR QUE A ETIQUETA DE CONJUNTO E "tcc3_ood_fusion"
REM -------------------------------------------------------------------
REM E a armadilha 1 das quatro de sobrescrita. Ja existem duas rodadas em
REM disco: "tcc3_30k" (a da dissertacao) e "tcc3_ood" (o benchmark dos
REM geradores modernos, split test do MESMO manifesto). Uma terceira
REM etiqueta mantem as tres lado a lado e torna a consolidacao capaz de
REM mirar exatamente nesta.
REM
REM -------------------------------------------------------------------
REM POR QUE OS DISPOSITIVOS SAO ESTES
REM -------------------------------------------------------------------
REM Identicos aos da rodada padrao e aos do benchmark: objeto-sombra em
REM GPU, campos e retas em CPU. Trocar CPU por GPU nos campos altera o
REM escore em ate 2,2e-02, e os escores desta rodada vao para a MESMA
REM fusao que consome os das outras -- precisam ser comparaveis.
REM
REM As retas EXIGEM CUDA_VISIBLE_DEVICES vazio: o PointNet dos autores
REM manda a matriz identidade para CUDA sempre que a placa estiver
REM visivel (lines_model.py:39-41). E a armadilha 11.
REM
REM -------------------------------------------------------------------
REM TEMPO
REM -------------------------------------------------------------------
REM Escalado da rodada do benchmark (14.788 imagens): objeto-sombra
REM ~3 min de GPU, campos ~14 min de CPU, retas ~8 min de CPU. Total em
REM sequencia ~25 min.
REM
REM Toda etapa e retomavel: itens ja gravados sao pulados.
REM
REM Codigo 1 no extrator de retas NAO significa rodada perdida -- ele
REM encerra com 1 quando qualquer imagem fica sem segmento detectado, o
REM que e falha legitima e esperada em fracao pequena do corpus.
REM
REM Uso:
REM   automacao\t04_glide_fusao.bat            -> as tres etapas
REM   automacao\t04_glide_fusao.bat sombra     -> so objeto-sombra
REM   automacao\t04_glide_fusao.bat campos     -> so campos
REM   automacao\t04_glide_fusao.bat retas      -> so retas
REM ===================================================================

@echo on
cd /d "%~dp0.."

set RAIZ=/mnt/c/Users/ferre/projects/tcc3-deteccao-imagens-ia
set MANIFESTO=%RAIZ%/data/manifesto30k_ood.csv
set CONJUNTO=tcc3_ood_fusion
set SPLIT=fusion
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
wsl -d Ubuntu-24.04 -u root -- bash -c "cd /root/SSIS/demo && /root/geo/bin/python %RAIZ%/automacao/wsl/extrair_object_shadow_ssis.py --manifesto %MANIFESTO% --split %SPLIT% --saida %RAIZ%/data/t04_object_shadow --conjunto %CONJUNTO%" >> logs\t04_glide_sombra.log 2>> logs\t04_glide_sombra.err
set COD=%ERRORLEVEL%
echo [%DATE% %TIME%] objeto-sombra: codigo %COD% >>logs\t04_glide_status.log
echo objeto-sombra: codigo %COD%
if not "%ETAPA%"=="todas" exit /b %COD%

:campos
echo === campos de perspectiva (CPU) ===
wsl -d Ubuntu-24.04 -u root -- bash -c "cd /root && CUDA_VISIBLE_DEVICES='' /root/geo/bin/python %RAIZ%/automacao/wsl/extrair_perspective_fields.py --manifesto %MANIFESTO% --split %SPLIT% --saida %RAIZ%/data/t04_perspective_fields --conjunto %CONJUNTO% --dispositivo cpu" >> logs\t04_glide_campos.log 2>> logs\t04_glide_campos.err
set COD=%ERRORLEVEL%
echo [%DATE% %TIME%] campos: codigo %COD% >>logs\t04_glide_status.log
echo campos: codigo %COD%
if not "%ETAPA%"=="todas" exit /b %COD%

:retas
echo === segmentos de reta (CPU) ===
wsl -d Ubuntu-24.04 -u root -- bash -c "cd /root/DeepLSD && CUDA_VISIBLE_DEVICES='' /root/geo/bin/python %RAIZ%/automacao/wsl/extrair_line_segments.py --manifesto %MANIFESTO% --split %SPLIT% --saida %RAIZ%/data/t04_line_segments --conjunto %CONJUNTO% --dispositivo cpu" >> logs\t04_glide_retas.log 2>> logs\t04_glide_retas.err
set COD=%ERRORLEVEL%
echo [%DATE% %TIME%] retas: codigo %COD% >>logs\t04_glide_status.log
echo retas: codigo %COD%

echo.
echo ===================================================================
echo Extracao concluida. Proximos passos, NESTA ordem:
echo.
echo   REM 1. classifica os mapas objeto-sombra desta rodada, e so dela
echo   python automacao\avaliar_t04_corpus.py --conjunto tcc3_ood_fusion ^
echo          --splits fusion --sufixo _ood_fusion
echo.
echo   REM 2. junta as tres representacoes desta rodada
echo   python automacao\consolidar_escores_t04.py --conjunto tcc3_ood_fusion ^
echo          --sufixo _ood_fusion --saida resultados\t04_escores_componentes_ood_fusion.csv
echo.
echo   REM 3. funde com as duas rodadas anteriores num arquivo que cubra
echo   REM    o protocolo OOD inteiro, e reajusta a T05
echo   python automacao\recalibrar_t05_fusion.py
echo ===================================================================
exit /b 0
