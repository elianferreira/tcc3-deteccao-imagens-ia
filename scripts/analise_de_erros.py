"""Analise de erros da Etapa 5.

A Secao 3.5.2 do TCC 2 preve, na Etapa 5, examinar "casos em que tecnicas
distintas discordam e imagens que todas as tecnicas classificam incorretamente,
com agrupamento por tipo de gerador".

Este script responde a quatro perguntas que a AUC agregada nao responde:

1. Quais imagens **todas** as tecnicas erram? Sao os casos irredutiveis com o
   conjunto atual de evidencias -- se concentrados em um gerador, indicam que
   aquele gerador nao deixa nenhum dos rastros procurados.
2. Onde as tecnicas **discordam**? Discordancia alta indica evidencias
   complementares, que e a premissa da fusao tardia (T05).
3. A fusao **aproveita** essa complementaridade? Compara-se quantos casos T05
   acerta entre os que cada tecnica isolada erra.
4. Ha gerador em que uma tecnica **inverte** -- erra mais que o acaso? A
   inversao e qualitativamente distinta da degradacao e precisa ser reportada
   separadamente.

Opera sobre os escores ja gravados pela campanha; nao reexecuta inferencia.

Uso::

    python scripts/analise_de_erros.py --protocol ood
"""

from __future__ import annotations

import argparse
import json
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import DATA_DIR, DECISION_THRESHOLD, RESULTS_DIR   # noqa: E402

PROB_DIR = RESULTS_DIR / "probabilidades"
TECNICAS = ("T01", "T02", "T03", "T05")


def carregar(protocolo: str, manifesto: Path) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    frame = pd.read_csv(manifesto)
    teste = frame[frame["split"] == "test"].reset_index(drop=True)

    probabilidades: dict[str, np.ndarray] = {}
    for tecnica in TECNICAS:
        caminho = PROB_DIR / f"{protocolo}__{tecnica}__clean.npy"
        if not caminho.exists():
            print(f"[aviso] {tecnica}: {caminho.name} ausente")
            continue
        vetor = np.load(caminho)
        if len(vetor) != len(teste):
            print(f"[aviso] {tecnica}: {len(vetor)} escores para {len(teste)} linhas; ignorado")
            continue
        probabilidades[tecnica] = vetor

    return teste, probabilidades


def main() -> int:
    parser = argparse.ArgumentParser(description="Analise de erros (Etapa 5)")
    parser.add_argument("--protocol", default="ood", choices=["standard", "ood"])
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--limiar", type=float, default=DECISION_THRESHOLD)
    parser.add_argument("--exemplos", type=int, default=15,
                        help="quantos caminhos exportar por categoria")
    args = parser.parse_args()

    manifesto = args.manifest or DATA_DIR / (
        "manifesto30k_ood.csv" if args.protocol == "ood" else "manifesto30k_standard.csv"
    )
    if not manifesto.exists():
        print(f"ERRO: manifesto nao encontrado em {manifesto}")
        return 1

    teste, probabilidades = carregar(args.protocol, manifesto)
    if not probabilidades:
        print("ERRO: nenhum vetor de probabilidades disponivel")
        return 1

    y = teste["label"].to_numpy().astype(int)
    geradores = teste["generator"].to_numpy()
    disponiveis = list(probabilidades)

    # Acerto por tecnica no limiar de decisao.
    acertos = {t: ((probabilidades[t] >= args.limiar).astype(int) == y)
               for t in disponiveis}
    isoladas = [t for t in disponiveis if t != "T05"]

    saida: dict = {
        "protocolo": args.protocol,
        "limiar": args.limiar,
        "n_teste": int(len(y)),
        "tecnicas": disponiveis,
    }

    print("=" * 74)
    print(f"ANALISE DE ERROS - protocolo {args.protocol} - {len(y)} imagens")
    print("=" * 74)

    # --- 1. Imagens que todas as tecnicas isoladas erram --------------------
    todas_erram = ~np.any([acertos[t] for t in isoladas], axis=0)
    print(f"\n1. IMAGENS QUE TODAS AS {len(isoladas)} TECNICAS ISOLADAS ERRAM")
    print(f"   total: {int(todas_erram.sum())} de {len(y)} "
          f"({todas_erram.mean() * 100:.1f}%)")

    por_gerador = []
    for gerador in sorted(set(geradores)):
        mascara = geradores == gerador
        n = int(mascara.sum())
        erradas = int((todas_erram & mascara).sum())
        por_gerador.append({
            "gerador": gerador, "n": n, "todas_erram": erradas,
            "taxa": erradas / n if n else 0.0,
        })
    por_gerador.sort(key=lambda d: d["taxa"], reverse=True)
    saida["todas_erram_por_gerador"] = por_gerador

    print(f"\n   {'gerador':<24} {'n':>6} {'todas erram':>12} {'taxa':>8}")
    for linha in por_gerador:
        print(f"   {linha['gerador']:<24} {linha['n']:>6} "
              f"{linha['todas_erram']:>12} {linha['taxa'] * 100:>7.1f}%")

    # --- 2. Discordancia entre tecnicas ------------------------------------
    print("\n2. DISCORDANCIA ENTRE PARES DE TECNICAS")
    print("   (fracao de imagens em que uma acerta e a outra erra)")
    discordancias = {}
    for a, b in combinations(isoladas, 2):
        taxa = float((acertos[a] != acertos[b]).mean())
        discordancias[f"{a}_{b}"] = taxa
        print(f"   {a} x {b}: {taxa * 100:.1f}%")
    saida["discordancia"] = discordancias

    # Quantas tecnicas acertam cada imagem.
    n_acertos = np.sum([acertos[t].astype(int) for t in isoladas], axis=0)
    distribuicao = {int(k): int(v) for k, v in zip(*np.unique(n_acertos, return_counts=True))}
    saida["distribuicao_de_acertos"] = distribuicao
    print("\n   imagens por numero de tecnicas que acertam:")
    for k in sorted(distribuicao):
        print(f"     {k} de {len(isoladas)}: {distribuicao[k]:>6} "
              f"({distribuicao[k] / len(y) * 100:.1f}%)")

    # --- 3. A fusao aproveita a complementaridade? -------------------------
    if "T05" in acertos:
        print("\n3. RESGATE PELA FUSAO (T05)")
        resgate = {}
        for tecnica in isoladas:
            erros = ~acertos[tecnica]
            if erros.sum() == 0:
                continue
            recuperados = int((acertos["T05"] & erros).sum())
            resgate[tecnica] = {
                "erros_da_tecnica": int(erros.sum()),
                "recuperados_por_T05": recuperados,
                "taxa": recuperados / int(erros.sum()),
            }
            print(f"   entre os {int(erros.sum())} erros de {tecnica}, "
                  f"T05 acerta {recuperados} ({recuperados / erros.sum() * 100:.1f}%)")

        # O inverso importa igualmente: casos que a fusao estraga.
        maioria = n_acertos >= (len(isoladas) + 1) // 2
        estragados = int((maioria & ~acertos["T05"]).sum())
        print(f"\n   casos em que a maioria acerta e T05 erra: {estragados} "
              f"({estragados / len(y) * 100:.1f}%)")
        resgate["estragados_pela_fusao"] = estragados
        saida["resgate_pela_fusao"] = resgate

    # --- 4. Inversao por gerador -------------------------------------------
    #
    # A inversao precisa ser medida por AUC, nao por taxa de acerto. Cada
    # subconjunto de gerador contem apenas imagens sinteticas, de modo que a
    # taxa de acerto ali equivale ao recall: um limiar conservador produz
    # recall baixo em TODOS os geradores sem que exista inversao alguma --
    # o classificador apenas exige mais evidencia para acusar.
    #
    # Inversao e outra coisa: e o classificador ordenar as sinteticas daquele
    # gerador como MAIS reais que as proprias imagens reais. Isso so aparece
    # comparando cada gerador contra o conjunto real, via AUC < 0,5.
    from sklearn.metrics import roc_auc_score

    print("\n4. INVERSAO - AUC do gerador contra as imagens reais")
    print("   (AUC < 0,50 significa ordenar sinteticas como mais reais que reais)")
    mascara_real = geradores == "real"
    inversoes, auc_por_gerador = [], []

    for tecnica in disponiveis:
        for gerador in sorted(set(geradores)):
            if gerador == "real":
                continue
            mascara = geradores == gerador
            if mascara.sum() < 30:
                continue
            selecao = mascara | mascara_real
            auc = float(roc_auc_score(y[selecao], probabilidades[tecnica][selecao]))
            recall = float(acertos[tecnica][mascara].mean())
            registro = {"tecnica": tecnica, "gerador": gerador, "auc": auc,
                        "recall": recall, "n": int(mascara.sum())}
            auc_por_gerador.append(registro)
            if auc < 0.5:
                inversoes.append(registro)

    inversoes.sort(key=lambda d: d["auc"])
    saida["inversoes"] = inversoes
    saida["auc_por_gerador"] = auc_por_gerador

    if inversoes:
        for linha in inversoes:
            print(f"   {linha['tecnica']} em {linha['gerador']:<22} "
                  f"AUC={linha['auc']:.4f}  (recall {linha['recall'] * 100:.1f}%)")
    else:
        print("   nenhuma")

    # Contraste explicito: recall baixo sem inversao e o caso comum.
    baixo_recall_sem_inversao = [
        r for r in auc_por_gerador if r["recall"] < 0.5 and r["auc"] >= 0.5
    ]
    print(f"\n   casos de recall < 50% SEM inversao: {len(baixo_recall_sem_inversao)}")
    print("   (limiar conservador, nao falha de ordenacao)")

    # --- 5. Exemplos para inspecao qualitativa -----------------------------
    exemplos: dict[str, list[str]] = {}
    indices_dificeis = np.where(todas_erram)[0][: args.exemplos]
    exemplos["todas_erram"] = [teste.loc[i, "path"] for i in indices_dificeis]

    if "T05" in acertos:
        so_fusao = np.where(acertos["T05"] & todas_erram)[0][: args.exemplos]
        exemplos["so_a_fusao_acerta"] = [teste.loc[i, "path"] for i in so_fusao]
    saida["exemplos"] = exemplos

    destino = RESULTS_DIR / f"analise_de_erros_{args.protocol}.json"
    destino.write_text(json.dumps(saida, indent=2, ensure_ascii=False), encoding="utf-8")

    csv_destino = RESULTS_DIR / f"analise_de_erros_{args.protocol}_por_gerador.csv"
    pd.DataFrame(por_gerador).to_csv(csv_destino, index=False)

    print("\n" + "=" * 74)
    print(f"Gravado em {destino}")
    print(f"Gravado em {csv_destino}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
