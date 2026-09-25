"""Qual classificador de fusao generaliza melhor: logistica, floresta ou boosting.

Continuacao de `mlp.py`, que ja descartou a MLP. Mesmo protocolo: particao de
ajuste, particao de avaliacao, vetor de atributos e `StandardScaler` identicos
-- **a unica variavel e o classificador**.

Por que a expectativa e desfavoravel as arvores
-----------------------------------------------
Arvores nao extrapolam. Fora da regiao coberta pelo ajuste elas devolvem o
valor da folha, constante. Aqui o ajuste ve **um gerador sintetico** (`glide`) e
a avaliacao ve **treze**, cujas distribuicoes de escore ficam em outro lugar --
a T02, por exemplo, tem mediana 1,0 no `stable_diffusion_1_4` e 2,6e-18 no
`flux`. Um modelo linear monotonico degrada suavemente sobre esse deslocamento;
uma floresta satura.

Isso e expectativa, nao resultado. O script mede.

`n_estimators=200` reproduz o que `codigo/configuracao.py` ja define para a
opcao `random_forest` do T05Fusion -- a comparacao e com a alternativa que o
projeto de fato ofereceria, nao com uma inventada aqui.
"""

from __future__ import annotations

import json
import sys
import warnings
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


def construtores():
    from sklearn.ensemble import (
        ExtraTreesClassifier, GradientBoostingClassifier, RandomForestClassifier,
    )
    from sklearn.linear_model import LogisticRegression

    return {
        "logistica": lambda s: LogisticRegression(max_iter=1000, random_state=s),
        "rf_200": lambda s: RandomForestClassifier(
            n_estimators=200, random_state=s, n_jobs=-1),
        "rf_200_prof3": lambda s: RandomForestClassifier(
            n_estimators=200, max_depth=3, random_state=s, n_jobs=-1),
        "extra_200": lambda s: ExtraTreesClassifier(
            n_estimators=200, random_state=s, n_jobs=-1),
        "boosting": lambda s: GradientBoostingClassifier(random_state=s),
    }


def main() -> int:
    warnings.filterwarnings("ignore")
    from sklearn.metrics import roc_auc_score
    from sklearn.preprocessing import StandardScaler

    ajuste = montar("fusion")
    ajuste = ajuste[ajuste["limpa"]].copy()
    avaliacao = montar("test")
    avaliacao = avaliacao[avaliacao["limpa"]].copy()
    print(f"ajuste {len(ajuste)}  avaliacao {len(avaliacao)}\n")

    linhas, previsoes = [], {}

    for rotulo_fontes, fontes in (("sem_T03", FONTES_SEM_T03),
                                  ("com_T03", FONTES_COM_T03)):
        X = vetorizar(ajuste, fontes, confiabilidade=False)
        y = ajuste["label"].to_numpy()
        Xa = vetorizar(avaliacao, fontes, confiabilidade=False)
        ya = avaliacao["label"].to_numpy()
        escalador = StandardScaler().fit(X)
        Xs, Xas = escalador.transform(X), escalador.transform(Xa)

        for rotulo, construir in construtores().items():
            ajustes, avals, soma = [], [], None
            for semente in SEMENTES:
                modelo = construir(semente).fit(Xs, y)
                ajustes.append(roc_auc_score(y, modelo.predict_proba(Xs)[:, 1]))
                p = modelo.predict_proba(Xas)[:, 1]
                avals.append(roc_auc_score(ya, p))
                soma = p if soma is None else soma + p
                if rotulo == "logistica":
                    break                      # determinista: uma basta
            nome = f"{rotulo_fontes}__{rotulo}"
            previsoes[nome] = soma / len(avals)
            linhas.append({
                "modelo": nome,
                "auc_ajuste": float(np.mean(ajustes)),
                "auc_aval": float(np.mean(avals)),
                "auc_aval_min": float(np.min(avals)),
                "auc_aval_max": float(np.max(avals)),
            })

    tabela = pd.DataFrame(linhas)
    tabela["excesso"] = tabela["auc_ajuste"] - tabela["auc_aval"]
    print(tabela.to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    print("\n--- DeLong contra a logistica, mesmo conjunto ---")
    ya = avaliacao["label"].to_numpy()
    for rotulo_fontes in ("sem_T03", "com_T03"):
        referencia = previsoes[f"{rotulo_fontes}__logistica"]
        for rotulo in construtores():
            if rotulo == "logistica":
                continue
            t = delong_test(ya, referencia, previsoes[f"{rotulo_fontes}__{rotulo}"])
            vencedor = "logistica" if t["difference"] > 0 else rotulo
            print(f"   {rotulo_fontes:8} logistica x {rotulo:13} "
                  f"dif {t['difference']:+.4f}  z {t['z']:+6.2f}   ganha: {vencedor}")

    melhor = tabela.loc[tabela["auc_aval"].idxmax(), "modelo"]
    baixo, alto = bootstrap_auc_ci(ya, previsoes[melhor], n_resamples=1000)
    print(f"\nmelhor: {melhor}  IC 95% [{baixo:.4f}; {alto:.4f}]")

    tabela.to_csv(SAIDA / "classificadores.csv", index=False)
    (SAIDA / "classificadores.json").write_text(
        json.dumps(linhas, indent=2), encoding="utf-8"
    )
    print(f"\ngravado em {SAIDA}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
