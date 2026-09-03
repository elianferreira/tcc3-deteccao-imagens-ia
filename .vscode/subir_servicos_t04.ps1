# Sobe os dois servicos residentes de T04 no WSL2.
#
# POR QUE DOIS SERVICOS, E NAO UM
# -------------------------------
# O PointNet dos autores (lines_model.py:39-41) manda a matriz identidade para
# CUDA sempre que torch.cuda.is_available() for verdadeiro -- nao quando o
# modelo esta na GPU. O contorno e CUDA_VISIBLE_DEVICES vazio, que vale para o
# processo inteiro e conflita com o objeto-sombra, que roda em GPU. Daí dois.
# (Armadilha 11 do docs/ESTADO_ATUAL.md.)
#
# POR QUE Start-Process, E NAO wsl ... &
# --------------------------------------
# Processo lancado de dentro do wsl.exe nao sobrevive ao retorno da chamada,
# nem com setsid nem com nohup: morre sem escrever no log. O que segura e
# lancar pelo lado Windows, mantendo um processo vivo que ancora o do WSL.
# (Armadilha 12.)
#
# OS DISPOSITIVOS SAO PARTE DO RESULTADO
# --------------------------------------
# Objeto-sombra em GPU, campos e retas em CPU -- iguais aos do lote que produziu
# os numeros do capitulo. Trocar CPU por GPU nos campos altera o escore em ate
# 2,2e-02, e os escores deixariam de ser comparaveis aos reportados.

$ErrorActionPreference = 'Stop'
$raiz = '/mnt/c/Users/ferre/projects/tcc3-deteccao-imagens-ia'
$py   = '/root/geo/bin/python'

$vivos = @(Get-NetTCPConnection -State Listen -ErrorAction SilentlyContinue |
           Where-Object { $_.LocalPort -in 8404, 8405 })
if ($vivos.Count -eq 2) {
    Write-Host 'Os dois servicos T04 ja estao no ar (8404 e 8405). Nada a fazer.' -ForegroundColor Green
    exit 0
}
if ($vivos.Count -eq 1) {
    Write-Host 'AVISO: so um dos servicos esta no ar. Pare os dois antes de subir de novo:' -ForegroundColor Yellow
    Write-Host '  Tarefa "T04: parar servicos"' -ForegroundColor Yellow
    exit 1
}

Write-Host '=== 8404: objeto-sombra em GPU ===' -ForegroundColor Cyan
Start-Process wsl.exe -ArgumentList @(
    '-d', 'Ubuntu-24.04', '-u', 'root', '--',
    $py, '-u', "$raiz/scripts/wsl/servico_t04.py",
    '--porta', '8404', '--dispositivo', 'cuda',
    '--representacoes', 'object_shadow'
) -WindowStyle Hidden

Write-Host '=== 8405: campos e retas em CPU, com a placa escondida ===' -ForegroundColor Cyan
Start-Process wsl.exe -ArgumentList @(
    '-d', 'Ubuntu-24.04', '-u', 'root', '--', 'env', 'CUDA_VISIBLE_DEVICES=',
    $py, '-u', "$raiz/scripts/wsl/servico_t04.py",
    '--porta', '8405', '--dispositivo', 'cpu',
    '--representacoes', 'perspective_fields,line_segment'
) -WindowStyle Hidden

Write-Host ''
Write-Host 'Carregando os modelos (~10 s cada). Aguardando as portas responderem...'
$limite = 90
for ($i = 0; $i -lt $limite; $i++) {
    Start-Sleep -Seconds 1
    $n = @(Get-NetTCPConnection -State Listen -ErrorAction SilentlyContinue |
           Where-Object { $_.LocalPort -in 8404, 8405 }).Count
    if ($n -eq 2) {
        Write-Host ''
        Write-Host "Servicos no ar em $($i + 1) s. Agora sim pode subir a interface." -ForegroundColor Green
        Write-Host '  Executar e Depurar -> "Interface (Gradio) - configuracao completa"'
        exit 0
    }
    Write-Host '.' -NoNewline
}

Write-Host ''
Write-Host "Os servicos nao responderam em $limite s." -ForegroundColor Red
Write-Host 'Confira com a tarefa "T04: conferir servicos". Causa mais comum: a VRAM'
Write-Host 'estava ocupada -- feche o que estiver usando a placa e tente de novo.'
exit 1
