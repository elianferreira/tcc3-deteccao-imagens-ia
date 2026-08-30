"""Remede o RNF01 pela interface, agora com T04 entrando pelo servico residente.

Ate 30/08/2026 T04 ficava fora da tela e o RNF01 media 17,871 s contra o teto de
30 s. Com o servico do WSL2 de pe, T04 passa a pontuar imagens arbitrarias, e o
custo dela entra na soma. Este script mede quanto.

Percorre exatamente o caminho da interface -- ``DetectionService.analyze`` --, e
nao os modulos isolados: e a soma que o RNF01 limita.

Uso::

    set TCC3_T04_SERVICO=1
    python scripts/medir_rnf01_com_t04.py --n 5
"""

from __future__ import annotations

import argparse
import csv
import statistics
import sys
import time
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ))

from app.gradio_app import DetectionService                # noqa: E402

LIMITE = 30.0                                             # RNF01
ORDEM = ("T01", "T02", "T03", "T04", "T05")


def main() -> int:
    parser = argparse.ArgumentParser(description="RNF01 com T04 na interface")
    parser.add_argument("--n", type=int, default=5)
    parser.add_argument("--split", default="test")
    parser.add_argument("--manifesto", type=Path,
                        default=RAIZ / "data" / "manifesto30k_standard.csv")
    args = parser.parse_args()

    print("carregando os modulos da interface...", flush=True)
    t0 = time.perf_counter()
    servico = DetectionService()
    print(f"carga: {time.perf_counter() - t0:.1f}s")
    print(f"modulos: {sorted(servico.techniques)}")
    print(f"fusao:   {servico.fusion_model_name or '(nenhuma)'}")
    if servico.load_errors:
        print(f"ausentes: {servico.load_errors}")
    print()

    with open(args.manifesto, newline="", encoding="utf-8") as arquivo:
        imagens = [Path(l["path"]) for l in csv.DictReader(arquivo)
                   if l["split"] == args.split][:args.n]

    por_tecnica: dict[str, list[float]] = {t: [] for t in ORDEM}
    totais: list[float] = []

    cabecalho = "imagem".ljust(20) + "".join(t.rjust(10) for t in ORDEM) + "total".rjust(11)
    print(cabecalho)
    print("-" * len(cabecalho))

    for caminho in imagens:
        inicio = time.perf_counter()
        resultado = servico.analyze(caminho)
        total = time.perf_counter() - inicio
        totais.append(total)

        linha = caminho.stem[:19].ljust(20)
        for tecnica in ORDEM:
            item = resultado.get(tecnica, {})
            if item.get("status") == "ok" and "elapsed" in item:
                por_tecnica[tecnica].append(item["elapsed"])
                linha += f"{item['elapsed']:9.3f} "
            else:
                linha += f"{item.get('status', '-')[:9]:>9} "
        linha += f"{total:10.3f}"
        print(linha, flush=True)

    print("-" * len(cabecalho))
    print("\nmedia por modulo (s/imagem):")
    for tecnica in ORDEM:
        amostras = por_tecnica[tecnica]
        if amostras:
            desvio = statistics.stdev(amostras) if len(amostras) > 1 else 0.0
            print(f"  {tecnica}  {statistics.mean(amostras):8.3f} +- {desvio:.3f}")
        else:
            print(f"  {tecnica}  {'nao medido':>8}")

    media = statistics.mean(totais)
    print(f"\nRNF01: {media:.3f} s/imagem  (teto {LIMITE} s)")
    print(f"margem: {LIMITE - media:.3f} s")
    if media < LIMITE:
        print("\nRNF01 CUMPRIDO com T04 na interface.")
        return 0
    print("\nRNF01 VIOLADO -- T04 nao pode entrar na tela assim.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
