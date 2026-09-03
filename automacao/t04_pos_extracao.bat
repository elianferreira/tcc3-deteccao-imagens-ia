@echo off
REM Encadeia o que vem depois da extracao dos mapas objeto-sombra:
REM
REM   1. avaliacao de T04 sobre os tres splits, com as tres variantes de pesos
REM   2. protocolo padrao com T04 como quarta fonte da fusao T05
REM
REM Ambos usam a GPU e devem rodar apos o fim de tcc3_t04_extracao.
REM O passo 2 nao retreina nada e nao sobrescreve pesos/t05_fusion.pkl --
REM ver o cabecalho de automacao/pipeline_t04_fusao.py.

cd /d "%~dp0.."

.venv\Scripts\python.exe -u automacao\avaliar_t04_corpus.py >> logs\t04_pos_extracao.log 2>> logs\t04_pos_extracao.err
echo ">>> avaliacao: codigo %ERRORLEVEL%" >> logs\t04_pos_extracao.log

.venv\Scripts\python.exe -u automacao\pipeline_t04_fusao.py >> logs\t04_pos_extracao.log 2>> logs\t04_pos_extracao.err
echo ">>> fusao: codigo %ERRORLEVEL%" >> logs\t04_pos_extracao.log
