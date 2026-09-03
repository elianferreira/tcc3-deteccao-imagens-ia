"""Confronta os modelos de fusao T05 sobre os mesmos escores de entrada.

Por que comparar assim
----------------------
E a armadilha 6 do ``docs/ESTADO_ATUAL.md``: *"comparar modelos ajustados em
ocasioes distintas mede o reajuste"*. Um confronto descuidado ja sugeriu ganho
significativo de T04 na fusao (p < 0,0001) que sumiu ao ajustar os dois modelos
na mesma validacao (p = 0,182).

Aqui o cuidado e outro, porque a pergunta e outra: os modelos **ja estao
ajustados** e o que se quer saber e como cada um decide. Por isso todos recebem
**exatamente a mesma matriz de escores** -- as mesmas quatro colunas, das mesmas
imagens. Nada e reajustado. A unica variavel e o modelo.

As duas bancadas
----------------
1. **Split ``test`` do protocolo padrao** (9.000 imagens), lido do cache em
   ``results/scores/standard__*``. E in-distribution, e serve para responder
   "o modelo recalibrado perde onde o atual e forte?".
2. **A varredura de pastas** (``results/varredura_pastas/varredura.csv``), que
   traz as quatro fontes por imagem em 19 pastas. **Quatro delas sao ruido
   procedural** e ficam de fora: ver a secao RETRATACAO do ESTADO_ATUAL. Restam
   15 pastas validas -- a COCO e os 14 geradores -- e e a bancada que responde
   "o modelo recalibrado acerta na tela?".

Uso::

    python scripts/comparar_modelos_t05.py
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.config import DECISION_THRESHOLD, WEIGHTS_DIR   # noqa: E402
from src.metrics import compute_metrics                  # noqa: E402
from src.techniques.t05_fusion import T05Fusion          # noqa: E402

# As quatro pastas de data/amostra, que sao ruido 1/f e nao fontes reais.
PASTAS_INVALIDAS = {"fodb", "imagenet", "open_images", "raise"}

MODELOS = {
    "variante B (3 fontes, OOD)": "t05_fusion.pkl",
    "quatro fontes (padrao/val)": "t05_fusion__quatro_fontes.pkl",
    "quatro fontes (OOD/fusion)": "t05_fusion__quatro_fontes_ood.pkl",
}


def carregar(nome_arquivo: str) -> T05Fusion | None:
    caminho = WEIGHTS_DIR / nome_arquivo
    if not caminho.exists():
        return None
    return T05Fusion().load(caminho)


def matriz_do_cache() -> tuple[np.ndarray, np.ndarray] | None:
    """Escores de T01-T04 no split ``test`` do protocolo padrao."""
    cache = ROOT / "results" / "scores"
    colunas = []
    for tecnica in ("T01", "T02", "T03", "T04"):
        arquivo = cache / f"standard__{tecnica}__{tecnica}__clean.npy"
        if not arquivo.exists():
            print(f"[bancada 1] ausente: {arquivo.name}")
            return None
        colunas.append(np.load(arquivo))

    import pandas as pd
    manifesto = pd.read_csv(ROOT / "data" / "manifesto30k_standard.csv")
    y = manifesto[manifesto["split"] == "test"]["label"].to_numpy()
    matriz = np.column_stack(colunas)
    if matriz.shape[0] != y.shape[0]:
        print(f"[bancada 1] formas incompativeis: {matriz.shape} e {y.shape}")
        return None
    return matriz, y


def matriz_da_varredura() -> tuple[np.ndarray, np.ndarray, list[str]] | None:
    caminho = ROOT / "results" / "varredura_pastas" / "varredura.csv"
    if not caminho.exists():
        print(f"[bancada 2] ausente: {caminho}")
        return None
    linhas, rotulos, pastas = [], [], []
    for r in csv.DictReader(caminho.open(newline="", encoding="utf-8")):
        if r["pasta"] in PASTAS_INVALIDAS:
            continue
        linhas.append([float(r["T01"]), float(r["T02"]), float(r["T03"]), float(r["T04"])])
        rotulos.append(int(r["rotulo"]))
        pastas.append(r["pasta"])
    return np.asarray(linhas), np.asarray(rotulos), pastas


def main() -> int:
    argparse.ArgumentParser(description=__doc__).parse_args()

    modelos = {rotulo: carregar(arq) for rotulo, arq in MODELOS.items()}
    disponiveis = {k: v for k, v in modelos.items() if v is not None}
    for rotulo, modelo in modelos.items():
        if modelo is None:
            print(f"[modelo] AUSENTE: {rotulo} ({MODELOS[rotulo]})")
    if not disponiveis:
        return 1
    print()

    # ---- bancada 1: in-distribution, do cache -------------------------
    bancada1 = matriz_do_cache()
    if bancada1 is not None:
        matriz, y = bancada1
        print("=" * 74)
        print(f"BANCADA 1 - split 'test' do protocolo padrao, n={len(y)} (in-distribution)")
        print("=" * 74)
        print(f"{'modelo':<30}{'AUC':>9}{'acuracia':>10}{'FPR':>9}{'FNR':>9}")
        for rotulo, modelo in disponiveis.items():
            p = modelo.predict_proba_scores(matriz)
            m = compute_metrics(y, p, threshold=DECISION_THRESHOLD)
            print(f"{rotulo:<30}{m['auc']:>9.4f}{m['accuracy']:>10.4f}"
                  f"{m['fpr']:>9.4f}{m['fnr']:>9.4f}")
        print()

    # ---- bancada 2: a varredura, so as 15 pastas validas --------------
    bancada2 = matriz_da_varredura()
    if bancada2 is not None:
        matriz, y, pastas = bancada2
        print("=" * 74)
        print(f"BANCADA 2 - varredura, {len(set(pastas))} pastas validas, n={len(y)} imagens")
        print("=" * 74)
        print(f"{'modelo':<30}{'pastas certas':>15}{'reais':>8}{'sinteticas':>12}")
        detalhe: dict[str, dict[str, float]] = {}
        for rotulo, modelo in disponiveis.items():
            p = modelo.predict_proba_scores(matriz)
            por_pasta: dict[str, list[float]] = {}
            for pasta, escore in zip(pastas, p):
                por_pasta.setdefault(pasta, []).append(escore)
            medianas = {k: float(np.median(v)) for k, v in por_pasta.items()}
            detalhe[rotulo] = medianas
            rotulo_pasta = {pasta: rot for pasta, rot in zip(pastas, y)}
            certas = sum(
                1 for pasta, med in medianas.items()
                if (med >= DECISION_THRESHOLD) == bool(rotulo_pasta[pasta])
            )
            reais = sum(1 for pasta, med in medianas.items()
                        if rotulo_pasta[pasta] == 0 and med < DECISION_THRESHOLD)
            sint = sum(1 for pasta, med in medianas.items()
                       if rotulo_pasta[pasta] == 1 and med >= DECISION_THRESHOLD)
            n_reais = sum(1 for pasta in medianas if rotulo_pasta[pasta] == 0)
            n_sint = len(medianas) - n_reais
            print(f"{rotulo:<30}{f'{certas}/{len(medianas)}':>15}"
                  f"{f'{reais}/{n_reais}':>8}{f'{sint}/{n_sint}':>12}")

        print(f"\n{'-' * 74}")
        print("Mediana do escore de sintese por pasta (%). Para 'fake' alto e certo.")
        print(f"{'-' * 74}")
        cabecalho = f"{'pasta':<24}" + "".join(f"{r.split(' (')[0]:>17}" for r in detalhe)
        print(cabecalho)
        rotulo_pasta = {pasta: rot for pasta, rot in zip(pastas, y)}
        for pasta in sorted(detalhe[next(iter(detalhe))],
                            key=lambda p: (rotulo_pasta[p], p)):
            linha = f"{('real/' if rotulo_pasta[pasta] == 0 else 'fake/') + pasta:<24}"
            for rotulo in detalhe:
                med = detalhe[rotulo][pasta] * 100
                erro = "*" if (med >= 50) != bool(rotulo_pasta[pasta]) else " "
                linha += f"{f'{med:.1f}{erro}':>17}"
            print(linha)
        print("\n* marca erro.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
