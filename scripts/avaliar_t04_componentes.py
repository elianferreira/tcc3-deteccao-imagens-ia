"""Mede T04 sobre o corpus deste trabalho, representacao por representacao.

Produz a tabela que o Capitulo 4 precisa: quanto vale cada uma das tres
representacoes geometricas isoladamente, e quanto vale a agregacao das tres --
que e o que a tecnica T04 de fato entrega.

Entrada: ``results/t04_escores_componentes.csv``, produzido por
``scripts/consolidar_escores_t04.py``.

Por que medir isoladamente
--------------------------
O artigo original reporta as tres separadamente, e a agregacao por media e
decisao deste trabalho (``t04_geometry.py``, ``aggregation="mean"``). Sem o
recorte por componente nao se sabe se um resultado agregado fraco vem de todas
as representacoes ou de uma que arrasta as outras -- e essa distincao muda a
leitura.

A agregacao usa ``nanmean``, de modo que uma imagem sem alguma representacao e
avaliada pelas que tem, em vez de descartada. A cobertura de cada componente e
reportada junto, porque um AUC sobre subconjunto menor nao e diretamente
comparavel aos demais.

Uso::

    python scripts/avaliar_t04_componentes.py --split test
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import RESULTS_DIR                # noqa: E402
from src.metrics import compute_metrics           # noqa: E402
from src.techniques.t04_geometry import COMPONENTS   # noqa: E402


def metricas_de(coluna: np.ndarray, rotulos: np.ndarray) -> dict | None:
    """Metricas sobre as amostras em que a coluna tem valor."""
    presente = ~np.isnan(coluna)
    if presente.sum() == 0:
        return None
    y, prob = rotulos[presente], coluna[presente]
    if len(np.unique(y)) < 2:
        return {"n": int(presente.sum()), "auc": float("nan"),
                "observacao": "uma unica classe no recorte"}

    metricas = compute_metrics(y, prob)
    metricas["cobertura"] = float(presente.mean())
    return metricas


def main() -> int:
    parser = argparse.ArgumentParser(description="T04 por representacao geometrica")
    parser.add_argument("--escores", type=Path,
                        default=RESULTS_DIR / "t04_escores_componentes.csv")
    parser.add_argument("--split", default="test")
    args = parser.parse_args()

    if not args.escores.exists():
        print(f"ERRO: {args.escores} nao encontrado")
        print("Rode antes: python scripts/consolidar_escores_t04.py")
        return 1

    tabela = pd.read_csv(args.escores)
    if args.split:
        tabela = tabela[tabela["split"] == args.split]
    if tabela.empty:
        print(f"ERRO: nenhuma linha para o split '{args.split}'")
        return 1

    rotulos = tabela["label"].to_numpy(dtype=int)
    presentes = [c for c in COMPONENTS if c in tabela.columns]

    print("=" * 78)
    print("T04 POR REPRESENTACAO GEOMETRICA - corpus de Corvi et al. (2024)")
    print("=" * 78)
    print(f"Escores ......... {args.escores.name}")
    print(f"Split ........... {args.split}")
    print(f"Amostras ........ {len(tabela)}")
    print("=" * 78)
    print(f"{'representacao':<22}{'n':>8}{'cobertura':>11}{'AUC':>9}{'acuracia':>10}{'FPR':>8}")

    saida: dict = {"split": args.split, "n_total": int(len(tabela)),
                   "componentes": {}, "agregado": None}

    for componente in presentes:
        coluna = tabela[componente].to_numpy(dtype=float)
        metricas = metricas_de(coluna, rotulos)
        if metricas is None:
            print(f"{componente:<22}{'sem escores':>8}")
            continue
        saida["componentes"][componente] = metricas
        print(f"{componente:<22}{metricas['n_samples']:>8.0f}"
              f"{metricas['cobertura']:>10.1%}{metricas['auc']:>9.4f}"
              f"{metricas['accuracy']:>10.4f}{metricas['fpr']:>8.4f}")

    # --- agregacao, que e o que T04 entrega -------------------------------
    matriz = tabela[presentes].to_numpy(dtype=float)
    with np.errstate(invalid="ignore"):
        agregado = np.nanmean(matriz, axis=1)

    validas = ~np.isnan(agregado)
    if validas.sum() and len(np.unique(rotulos[validas])) == 2:
        metricas = compute_metrics(rotulos[validas], agregado[validas])
        metricas["cobertura"] = float(validas.mean())
        metricas["n_representacoes"] = len(presentes)
        saida["agregado"] = metricas
        print("-" * 78)
        print(f"{'T04 (media das ' + str(len(presentes)) + ')':<22}"
              f"{metricas['n_samples']:>8.0f}{metricas['cobertura']:>10.1%}"
              f"{metricas['auc']:>9.4f}{metricas['accuracy']:>10.4f}"
              f"{metricas['fpr']:>8.4f}")

    # Quantas imagens tem as tres representacoes ao mesmo tempo.
    completas = (~np.isnan(matriz)).all(axis=1).sum()
    print("=" * 78)
    print(f"Imagens com as {len(presentes)} representacoes: {completas} "
          f"({completas / len(tabela):.1%})")
    saida["n_com_todas"] = int(completas)

    destino = RESULTS_DIR / f"t04_componentes_{args.split}.json"
    destino.write_text(json.dumps(saida, indent=2, ensure_ascii=False), encoding="utf-8")

    linhas = [{"representacao": nome, "n": m.get("n_samples"),
               "cobertura": m.get("cobertura"), "auc": m.get("auc"),
               "acuracia": m.get("accuracy"), "fpr": m.get("fpr")}
              for nome, m in saida["componentes"].items()]
    if saida["agregado"]:
        linhas.append({"representacao": "T04 (agregado)",
                       "n": saida["agregado"]["n_samples"],
                       "cobertura": saida["agregado"]["cobertura"],
                       "auc": saida["agregado"]["auc"],
                       "acuracia": saida["agregado"]["accuracy"],
                       "fpr": saida["agregado"]["fpr"]})
    pd.DataFrame(linhas).to_csv(
        RESULTS_DIR / f"t04_componentes_{args.split}.csv", index=False)

    print(f"Gravado em {destino.name}")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
