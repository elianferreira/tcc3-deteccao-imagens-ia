"""Confere se o servico residente de T04 reproduz os escores da extracao em lote.

Por que este teste existe
-------------------------
O servico (``scripts/wsl/servico_t04.py``) recarrega os mesmos modelos e chama
as mesmas funcoes, mas por um caminho diferente: em lote os mapas objeto-sombra
passam pelo disco em JPEG q95, e aqui passam por um buffer em memoria. Se a
reproducao do round-trip estiver errada, os numeros da interface divergiriam dos
do Capitulo 4 -- silenciosamente, porque ambos seriam "plausiveis".

Compara contra ``results/t04_escores_componentes.csv``, que e a fonte dos valores
publicados (AUC 0,5384 / 0,5355 / 0,5187).

Uso::

    python scripts/verificar_paridade_servico_t04.py --n 30
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import urllib.error
import urllib.request
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent
ESCORES = RAIZ / "results" / "t04_escores_componentes.csv"
MANIFESTO = RAIZ / "data" / "manifesto30k_standard.csv"
# Dois processos, porque o lote usou dispositivos diferentes por representacao
# e o PointNet exige CUDA_VISIBLE_DEVICES="" (ver o cabecalho de servico_t04.py).
SERVICOS = {
    "http://127.0.0.1:8404": ("object_shadow",),
    "http://127.0.0.1:8405": ("perspective_fields", "line_segment"),
}
REPRESENTACOES = ("object_shadow", "perspective_fields", "line_segment")


def para_wsl(bruto: str) -> str:
    """``C:\\Users\\...`` -> ``/mnt/c/Users/...``, como os extratores fazem."""
    texto = bruto.strip().replace("\\", "/")
    if len(texto) > 1 and texto[1] == ":":
        return f"/mnt/{texto[0].lower()}{texto[2:]}"
    return texto


def pontuar(caminho_wsl: str, tempo_limite: int) -> dict:
    """Consulta os dois servicos e funde as respostas em um dicionario so."""
    fundido: dict = {}
    for base in SERVICOS:
        pedido = urllib.request.Request(
            f"{base}/escore",
            data=json.dumps({"caminho": caminho_wsl}).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST")
        with urllib.request.urlopen(pedido, timeout=tempo_limite) as resposta:
            fundido.update(json.loads(resposta.read()))
    return fundido


def main() -> int:
    parser = argparse.ArgumentParser(description="Paridade servico x lote (T04)")
    parser.add_argument("--n", type=int, default=30, help="imagens a conferir")
    parser.add_argument("--split", default="test")
    # 1e-3 e o teto do residuo conhecido do objeto-sombra: cuDNN escolhe
    # algoritmos de convolucao diferentes para lote 1 (servico) e lote 128
    # (avaliar_t04_corpus.py), o que mede 2,3e-04. Campos e retas ficam em
    # 1e-16. Uma divergencia acima deste teto e defeito, nao aritmetica.
    parser.add_argument("--tolerancia", type=float, default=1e-3,
                        help="diferenca absoluta aceita por representacao")
    parser.add_argument("--tempo-limite", type=int, default=300)
    args = parser.parse_args()

    for base in SERVICOS:
        try:
            with urllib.request.urlopen(f"{base}/saude", timeout=5) as r:
                saude = json.loads(r.read())
            print(f"{base} -> {saude['representacoes']} em {saude['dispositivo']} "
                  f"(cuda visivel: {saude['cuda_visivel']})")
        except (urllib.error.URLError, TimeoutError) as erro:
            print(f"ERRO: servico nao responde em {base} ({erro})")
            return 1
    print()

    # Caminho de cada imagem, pelo nome usado no CSV de escores.
    caminho_por_nome: dict[str, str] = {}
    with open(MANIFESTO, newline="", encoding="utf-8") as arquivo:
        for linha in csv.DictReader(arquivo):
            if linha["split"] == args.split:
                caminho_por_nome[Path(linha["path"]).stem] = linha["path"]

    esperados = []
    with open(ESCORES, newline="", encoding="utf-8") as arquivo:
        for linha in csv.DictReader(arquivo):
            if linha["split"] == args.split and linha["arquivo"] in caminho_por_nome:
                esperados.append(linha)
            if len(esperados) >= args.n:
                break

    if not esperados:
        print("ERRO: nenhuma imagem em comum entre o CSV de escores e o manifesto")
        return 1

    print(f"{'imagem':22} {'representacao':22} {'lote':>10} {'servico':>10} {'dif':>11}")
    print("-" * 80)

    divergentes = 0
    conferidas = 0
    falhas = 0
    piores: dict[str, float] = {r: 0.0 for r in REPRESENTACOES}

    for linha in esperados:
        nome = linha["arquivo"]
        try:
            obtido = pontuar(para_wsl(caminho_por_nome[nome]), args.tempo_limite)
        except Exception as erro:                          # noqa: BLE001
            print(f"{nome:22} FALHA NA CHAMADA: {type(erro).__name__}: {erro}")
            falhas += 1
            continue

        for representacao in REPRESENTACOES:
            bruto = linha.get(representacao, "")
            if bruto in ("", "nan", None):
                continue                                   # nao medida em lote
            esperado = float(bruto)
            atual = obtido.get(representacao)
            if atual is None:
                print(f"{nome:22} {representacao:22} {esperado:10.6f} {'ausente':>10} "
                      f"{'-':>11}  <-- servico nao produziu")
                divergentes += 1
                continue

            diferenca = abs(atual - esperado)
            piores[representacao] = max(piores[representacao], diferenca)
            conferidas += 1
            if diferenca > args.tolerancia:
                divergentes += 1
                print(f"{nome:22} {representacao:22} {esperado:10.6f} {atual:10.6f} "
                      f"{diferenca:11.2e}  <-- DIVERGE")

    print("-" * 80)
    print(f"comparacoes: {conferidas}   divergentes: {divergentes}   "
          f"falhas de chamada: {falhas}")
    print(f"tolerancia: {args.tolerancia:g}\n")
    print("maior diferenca por representacao:")
    for representacao, valor in piores.items():
        marca = "ok" if valor <= args.tolerancia else "DIVERGE"
        print(f"  {representacao:22} {valor:11.2e}   {marca}")

    if divergentes == 0 and falhas == 0:
        print("\nPARIDADE CONFIRMADA: o servico reproduz os escores da dissertacao.")
        return 0
    print("\nPARIDADE NAO CONFIRMADA -- nao usar na interface antes de resolver.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
