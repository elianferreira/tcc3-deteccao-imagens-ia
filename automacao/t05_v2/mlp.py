"""A T05 como MLP, contra a regressao logistica -- mesmo dado, mesma medida.

Sugestao de orientacao (08/09/2026): usar uma MLP no lugar da regressao
logistica na fusao. Este script decide por medicao em vez de argumento.

O que e mantido identico entre os dois lados
--------------------------------------------
Particao de ajuste, particao de avaliacao, vetor de atributos, `StandardScaler`
e o filtro de vazamento da replica. **A unica variavel e o classificador** --
e a comparacao so vale assim (armadilha 6 de ESTADO_ATUAL.md).

Por que varias sementes
-----------------------
A regressao logistica com estes dados e determinista; a MLP nao e -- ela
depende da inicializacao dos pesos. Reportar uma execucao seria escolher a
semente depois de ver o resultado. Cada arquitetura roda com cinco sementes e o
que entra na tabela e a media com o intervalo observado.

A pergunta de fundo
-------------------
Nao e "qual ajusta melhor a particao de ajuste" -- MLP ganha isso por
construcao, e a coluna `AUC ajuste` mostra por quanto. E "qual generaliza para
13 geradores nao vistos", que e a coluna `AUC avaliacao`. A distancia entre as
duas colunas e o custo da capacidade extra.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

RAIZ = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(RAIZ))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from ajustar import FONTES_SEM_T03, FONTES_COM_T03, montar, vetorizar  # noqa: E402
from codigo.metricas import bootstrap_auc_ci, delong_test  # noqa: E402

SAIDA = RAIZ / "resultados" / "t05_v2"
SEMENTES = (42, 123, 456, 7, 2026)

ARQUITETURAS = {
    "mlp_8": (8,),
    "mlp_16": (16,),
    "mlp_32_16": (32, 16),
}


def main() -> int:
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.neural_network import MLPClassifier
    from sklearn.preprocessing import StandardScaler

    ajuste = montar("fusion")
    ajuste = ajuste[ajuste["limpa"]].copy()
    avaliacao = montar("test")
    avaliacao = avaliacao[avaliacao["limpa"]].copy()
    print(f"ajuste {len(ajuste)}  avaliacao {len(avaliacao)}\n")

    linhas = []
    previsoes: dict[str, np.ndarray] = {}

    for rotulo_fontes, fontes in (("sem_T03", FONTES_SEM_T03),
                                  ("com_T03", FONTES_COM_T03)):
        X = vetorizar(ajuste, fontes, confiabilidade=False)
        y = ajuste["label"].to_numpy()
        Xa = vetorizar(avaliacao, fontes, confiabilidade=False)
        ya = avaliacao["label"].to_numpy()

        escalador = StandardScaler().fit(X)
        Xs, Xas = escalador.transform(X), escalador.transform(Xa)

        # --- referencia: a logistica que venceu o passo 3 --------------------
        base = LogisticRegression(max_iter=1000, random_state=42).fit(Xs, y)
        p_ajuste = base.predict_proba(Xs)[:, 1]
        p_aval = base.predict_proba(Xas)[:, 1]
        nome = f"{rotulo_fontes}__logistica"
        previsoes[nome] = p_aval
        linhas.append({
            "modelo": nome, "n_param": int(Xs.shape[1] + 1),
            "auc_ajuste": roc_auc_score(y, p_ajuste),
            "auc_aval": roc_auc_score(ya, p_aval),
            "auc_aval_min": np.nan, "auc_aval_max": np.nan,
        })

        # --- as MLP, cinco sementes cada -------------------------------------
        for rotulo_arq, camadas in ARQUITETURAS.items():
            ajustes, avals, ultimas = [], [], None
            for semente in SEMENTES:
                rede = MLPClassifier(
                    hidden_layer_sizes=camadas, max_iter=2000,
                    random_state=semente, early_stopping=False,
                )
                rede.fit(Xs, y)
                ajustes.append(roc_auc_score(y, rede.predict_proba(Xs)[:, 1]))
                pa = rede.predict_proba(Xas)[:, 1]
                avals.append(roc_auc_score(ya, pa))
                ultimas = pa if ultimas is None else ultimas + pa
            nome = f"{rotulo_fontes}__{rotulo_arq}"
            # A previsao guardada para o DeLong e a media das sementes: comparar
            # uma semente escolhida a dedo contra a logistica seria trapaca.
            previsoes[nome] = ultimas / len(SEMENTES)
            n_param = sum(
                a * b for a, b in zip((Xs.shape[1],) + camadas, camadas + (1,))
            ) + sum(camadas) + 1
            linhas.append({
                "modelo": nome, "n_param": int(n_param),
                "auc_ajuste": float(np.mean(ajustes)),
                "auc_aval": float(np.mean(avals)),
                "auc_aval_min": float(np.min(avals)),
                "auc_aval_max": float(np.max(avals)),
            })

    tabela = pd.DataFrame(linhas)
    tabela["excesso"] = tabela["auc_ajuste"] - tabela["auc_aval"]
    print(tabela.to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    print("\n--- DeLong contra a logistica, no mesmo conjunto ---")
    ya = avaliacao["label"].to_numpy()
    for rotulo_fontes in ("sem_T03", "com_T03"):
        referencia = previsoes[f"{rotulo_fontes}__logistica"]
        for rotulo_arq in ARQUITETURAS:
            nome = f"{rotulo_fontes}__{rotulo_arq}"
            t = delong_test(ya, referencia, previsoes[nome])
            marca = "logistica ganha" if t["difference"] > 0 else "MLP ganha"
            print(f"   {rotulo_fontes:8} logistica x {rotulo_arq:9} "
                  f"dif {t['difference']:+.4f}  p {t['p_value']:.4g}   {marca}")

    melhor = tabela.loc[tabela["auc_aval"].idxmax(), "modelo"]
    baixo, alto = bootstrap_auc_ci(ya, previsoes[melhor], n_resamples=1000)
    print(f"\nmelhor na avaliacao: {melhor}  IC 95% [{baixo:.4f}; {alto:.4f}]")

    tabela.to_csv(SAIDA / "mlp_vs_logistica.csv", index=False)
    (SAIDA / "mlp_vs_logistica.json").write_text(
        json.dumps(linhas, indent=2), encoding="utf-8"
    )
    print(f"\ngravado em {SAIDA}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
