@echo off
REM Extracao dos segmentos de reta de T04, via DeepLSD no WSL2.
REM
REM Roda em CPU de proposito: assim convive com o treino de T01 no protocolo
REM OOD por familias, que ocupa a GPU por horas. CUDA_VISIBLE_DEVICES vazio e
REM obrigatorio, e nao apenas uma preferencia -- o PointNet oficial de T04
REM manda a matriz identidade para CUDA sempre que CUDA esta disponivel
REM (lines_model.py:39-41), de modo que o caminho de CPU so funciona com a
REM placa escondida. Ver o obstaculo 11 em docs/T04_AMBIENTE_WSL2.md.
REM
REM Cerca de 0,6 s por imagem; ~3 h para teste e validacao.

cd /d "%~dp0.."

for %%S in (test val) do (
  wsl -d Ubuntu-24.04 -u root -- bash -c "cd /root/DeepLSD && CUDA_VISIBLE_DEVICES='' /root/geo/bin/python /mnt/c/Users/ferre/projects/tcc3-deteccao-imagens-ia/scripts/wsl/extrair_line_segments.py --manifesto /mnt/c/Users/ferre/projects/tcc3-deteccao-imagens-ia/data/manifesto30k_standard.csv --split %%S --saida /mnt/c/Users/ferre/projects/tcc3-deteccao-imagens-ia/data/t04_line_segments --conjunto tcc3_30k --dispositivo cpu" >> logs\t04_linhas.log 2>> logs\t04_linhas.err
  echo ">>> split %%S: codigo %%ERRORLEVEL%%" >> logs\t04_linhas.log
)
