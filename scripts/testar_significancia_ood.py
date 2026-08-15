"""Significancia estatistica das diferencas de AUC no protocolo OOD.

Responde duas perguntas que a tabela de AUC sozinha nao responde:

1. A vantagem de T05 sobre a melhor tecnica isolada excede a variabilidade
   amostral? (teste de DeLong, para AUCs correlacionadas sobre o mesmo conjunto)
2. Qual o intervalo de confianca de cada AUC? (bootstrap estratificado)

Sem isso, uma diferenca de 1 p.p. entre T05 e T01 nao pode ser afirmada como
superioridade -- e a contribuicao original do trabalho depende justamente dessa
comparacao.

Uso::

    python scripts/testar_significancia_ood.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import DATA_DIR, RESULTS_DIR          # noqa: E402
from src.metrics import bootstrap_auc_ci, delong_test  # noqa: E402

PROB_DIR = RESULTS_DIR / "probabilidades"
TECNICAS = ("T01", "T02", "T03", "T05")


def carregar_rotulos(manifesto: Path) -> np.ndarray:
    """Rotulos da particao de teste, na mesma ordem usada na inferencia."""
    frame = pd.read_csv(manifesto)
    teste = frame[frame["split"] == "test"]
    return teste["label"].to_numpy().astype(int)


def main() -> int:
    manifesto = DATA_DIR / "manifesto30k_ood.csv"
    if not manifesto.exists():
        print(f"ERRO: manifesto nao encontrado em {manifesto}")
        return 1

    y = carregar_rotulos(manifesto)
    print(f"Rotulos de teste: {len(y)} ({int((y == 1).sum())} sinteticas, "
          f"{int((y == 0).sum())} reais)\n")

    probabilidades: dict[str, np.ndarray] = {}
    for tecnica in TECNICAS:
        caminho = PROB_DIR / f"ood__{tecnica}__clean.npy"
        if not caminho.exists():
            print(f"[aviso] {tecnica}: {caminho.name} ausente")
            continue
        vetor = np.load(caminho)
        if len(vetor) != len(y):
            print(f"[aviso] {tecnica}: {len(vetor)} escores para {len(y)} rotulos; ignorado")
            continue
        probabilidades[tecnica] = vetor

    # As duas calibracoes de T05 sao testadas separadamente: elas produzem AUCs
    # distintas e a conclusao sobre a contribuicao do trabalho nao pode depender
    # de qual delas ficou por ultimo em disco.
    for rotulo in ("A", "B"):
        caminho = PROB_DIR / f"ood__T05__clean__variante_{rotulo}.npy"
        if caminho.exists():
            vetor = np.load(caminho)
            if len(vetor) == len(y):
                probabilidades[f"T05_{rotulo}"] = vetor

    if "T05" not in probabilidades:
        print("ERRO: probabilidades de T05 ausentes")
        return 1

    saida: dict = {"n_amostras": int(len(y)), "ic_auc": {}, "delong": {}}

    print("=" * 68)
    print("INTERVALO DE CONFIANCA DA AUC (bootstrap estratificado, 95%)")
    print("=" * 68)
    for tecnica, prob in probabilidades.items():
        baixo, alto = bootstrap_auc_ci(y, prob)
        saida["ic_auc"][tecnica] = [float(baixo), float(alto)]
        print(f"  {tecnica}: [{baixo:.4f}, {alto:.4f}]")

    print("\n" + "=" * 68)
    print("TESTE DE DELONG - T05 contra cada tecnica isolada")
    print("=" * 68)
    isoladas = [t for t in ("T01", "T02", "T03") if t in probabilidades]
    fusoes = [t for t in probabilidades if t.startswith("T05")]

    for fusao in fusoes:
        for tecnica in isoladas:
            resultado = delong_test(y, probabilidades[fusao], probabilidades[tecnica])
            saida["delong"][f"{fusao}_vs_{tecnica}"] = {
                chave: float(valor) for chave, valor in resultado.items()
            }
            p = resultado["p_value"]
            marca = "significativa" if p < 0.05 else "NAO significativa"
            print(f"  {fusao:<7} vs {tecnica}: diferenca {resultado['difference']:+.4f}, "
                  f"z = {resultado['z']:+.3f}, p = {p:.4g}  ->  {marca}")
        print()

    destino = RESULTS_DIR / "significancia_ood.json"
    destino.write_text(json.dumps(saida, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nGravado em {destino}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
