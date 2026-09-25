"""Diversidade de geradores e ausencia de fonte no ajuste da T05.

Dois experimentos, ambos sobre **escores em cache**: nenhuma imagem e
processada, nenhum peso de `pesos/` e tocado, nenhum `.npy` do cache e
reescrito.

Item 1 -- curva de diversidade da fusao (deixar-um-gerador-de-fora)
------------------------------------------------------------------
A particao `fusion` do protocolo OOD rigoroso, onde a variante B da T05 e
calibrada, tem **um unico gerador sintetico** (`glide`, 1.000 imagens, contra
1.350 reais). E o mesmo defeito de protocolo que `EXPERIMENTOS.md` da replicacao
mediu na T01 -- so que um nivel acima, na fusao.

Aqui a pergunta e medida direto: **a fusao precisa de diversidade de geradores no
conjunto de calibracao?** Para cada gerador `g` da particao `val` do protocolo
por familias, ajusta-se a T05 sobre `k` outros geradores e avalia-se em `g`, que
a fusao nunca viu. Volume de sinteticas constante (150 por gerador) para que a
unica variavel seja a diversidade -- mesmo controle do desenho da T01.

Item 2 -- dropout de fontes
---------------------------
Nos cinco modelos de `pesos/`, os quatro coeficientes da mascara de
disponibilidade sao exatamente 0,000: no ajuste nenhuma fonte faltou, a coluna
fica constante e o `StandardScaler` a zera. A RN07/RNF04 esta implementada e nao
treinada -- o comportamento com uma tecnica fora do ar e extrapolacao.

Aqui compara-se o ajuste padrao com um ajuste que sorteia fontes ausentes,
avaliando os dois com todas as fontes e com cada uma removida.

Origem dos dados (confirmada por reconstrucao de AUC publicada)
---------------------------------------------------------------
`resultados/scores/ood__T0x__T0x__{val,clean}.npy` sao do protocolo **por
familias** -- `data/manifesto30k_ood_familias.csv`, val 10.500 e test 7.138. O
prefixo `ood__` e compartilhado com o protocolo rigoroso e so o tamanho
desambigua: e a Armadilha 7. O passo 0 confere o alinhamento contra a tabela da
Secao 3.4 de `documentacao/RESULTADOS.md` antes de qualquer ajuste.

Uso::

    python automacao/experimento_fusao_diversidade.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ))

from codigo.tecnicas.t05_fusao.tecnica import T05Fusion, build_score_matrix  # noqa: E402

MANIFESTO = RAIZ / "data" / "manifesto30k_ood_familias.csv"
CACHE = RAIZ / "resultados" / "scores"
SAIDA = RAIZ / "resultados" / "fusao_diversidade"
FONTES = ("T01", "T02", "T03")          # T04 nao tem escore fora da particao `fusion`
SEMENTE = 42
POR_GERADOR = 150                       # teto de volume: o menor gerador da `val`

# Secao 3.4 de documentacao/RESULTADOS.md -- o alvo do passo 0.
AUC_PUBLICADA = {"T01": 0.9463, "T02": 0.6794, "T03": 0.6956}


def carregar(split: str, condicao: str) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    """Le a particao do manifesto e os escores em cache, na mesma ordem.

    `read_manifest` preserva a ordem do CSV filtrando por `split`, e
    `runner._infer` grava o `.npy` nessa ordem -- e por isso que a conferencia
    de forma abaixo e suficiente para garantir alinhamento posicional.
    """
    d = pd.read_csv(MANIFESTO)
    d = d[d.split == split].reset_index(drop=True)
    escores = {}
    for t in FONTES:
        caminho = CACHE / f"ood__{t}__{t}__{condicao}.npy"
        a = np.load(caminho)
        if a.shape != (len(d),):
            raise SystemExit(
                f"desalinhamento: {caminho.name} tem {a.shape[0]} posicoes, "
                f"a particao '{split}' tem {len(d)} linhas"
            )
        escores[t] = a
    return d, escores


def tpr_em_fpr(y: np.ndarray, p: np.ndarray, fpr_alvo: float) -> float:
    """Deteccao no ponto de operacao -- a AUC nao enxerga ponto de operacao."""
    limiar = np.quantile(p[y == 0], 1.0 - fpr_alvo)
    return float((p[y == 1] > limiar).mean())


def ajustar(matriz: np.ndarray, y: np.ndarray, dropout: float = 0.0,
            semente: int = SEMENTE) -> T05Fusion:
    """Ajusta a T05 do repositorio, opcionalmente com fontes ausentes sorteadas.

    O dropout entra como NaN na matriz de entrada -- exatamente o que
    `build_score_matrix` produz quando um modulo falha de verdade, entao o
    modelo aprende no mesmo formato em que vai receber a falha.
    """
    if dropout > 0.0:
        rng = np.random.default_rng(semente)
        matriz = matriz.copy()
        bloco = matriz[:, : len(FONTES)]
        mascara = rng.random(bloco.shape) < dropout
        # Nunca apagar todas as fontes de uma amostra: sem nenhuma evidencia
        # nao ha decisao a aprender, so o vies da prevalencia.
        mascara[mascara.all(axis=1)] = False
        bloco[mascara] = np.nan
        matriz[:, : len(FONTES)] = bloco
    return T05Fusion(seed=semente).fit_scores(matriz, y)


def matriz_de(escores: dict[str, np.ndarray], idx: np.ndarray, n: int) -> np.ndarray:
    return build_score_matrix({t: escores[t][idx] for t in FONTES}, n)


# ----------------------------------------------------------------------
# Passo 0 -- alinhamento
# ----------------------------------------------------------------------

def passo0() -> tuple[pd.DataFrame, dict, pd.DataFrame, dict]:
    print("=" * 72)
    print("PASSO 0 -- alinhamento dos escores em cache")
    print("=" * 72)
    dv, ev = carregar("val", "val")
    dt, et = carregar("test", "clean")
    print(f"val  {len(dv):>6} linhas, {dv.generator.nunique() - 1} geradores sinteticos")
    print(f"test {len(dt):>6} linhas, {dt.generator.nunique() - 1} geradores sinteticos")
    print("\nAUC reconstruida na `test` x tabela da Secao 3.4 de RESULTADOS.md:")
    ok = True
    for t in FONTES:
        auc = roc_auc_score(dt.label.values, et[t])
        alvo = AUC_PUBLICADA[t]
        bate = abs(auc - alvo) < 5e-4
        ok &= bate
        print(f"  {t}  medido {auc:.4f}  publicado {alvo:.4f}  {'OK' if bate else 'DIVERGE'}")
    if not ok:
        raise SystemExit("alinhamento reprovado -- nao seguir com os ajustes")
    print("\nAlinhamento confirmado. Os escores em cache sao os do protocolo por familias.")
    return dv, ev, dt, et


# ----------------------------------------------------------------------
# Item 1 -- curva de diversidade da fusao
# ----------------------------------------------------------------------

def item1(dv: pd.DataFrame, ev: dict) -> pd.DataFrame:
    print("\n" + "=" * 72)
    print("ITEM 1 -- diversidade de geradores no ajuste da fusao (deixar-um-de-fora)")
    print("=" * 72)
    rng = np.random.default_rng(SEMENTE)

    reais = dv.index[dv.label == 0].to_numpy()
    rng.shuffle(reais)
    meio = len(reais) // 2
    reais_ajuste, reais_aval = reais[:meio], reais[meio:]
    print(f"reais: {len(reais_ajuste)} para ajuste, {len(reais_aval)} para avaliacao "
          f"(disjuntos -- nenhuma real aparece nos dois lados)")

    geradores = sorted(g for g in dv.generator.unique() if g != "real")
    # Volume constante por gerador: a variavel medida e diversidade, nao volume.
    amostra = {}
    for g in geradores:
        idx = dv.index[dv.generator == g].to_numpy()
        rng.shuffle(idx)
        amostra[g] = idx[:POR_GERADOR]
    print(f"{len(geradores)} geradores, {POR_GERADOR} sinteticas de cada no ajuste\n")

    linhas = []
    for fora in geradores:
        disponiveis = [g for g in geradores if g != fora]
        idx_aval = np.concatenate([reais_aval, dv.index[dv.generator == fora].to_numpy()])
        y_aval = dv.label.values[idx_aval]
        m_aval = matriz_de(ev, idx_aval, len(idx_aval))

        for t in FONTES:                                   # componentes isolados
            linhas.append(dict(fora=fora, modelo=t, k=np.nan, repeticao=0,
                               auc=roc_auc_score(y_aval, ev[t][idx_aval]),
                               tpr5=tpr_em_fpr(y_aval, ev[t][idx_aval], 0.05)))

        for k in (1, 3, 6, 10):
            for rep in range(5):
                escolhidos = rng.choice(disponiveis, size=k, replace=False)
                idx_sin = np.concatenate([amostra[g] for g in escolhidos])
                n_reais = min(len(idx_sin), len(reais_ajuste))
                idx_aj = np.concatenate([rng.choice(reais_ajuste, n_reais, replace=False), idx_sin])
                modelo = ajustar(matriz_de(ev, idx_aj, len(idx_aj)), dv.label.values[idx_aj],
                                 semente=SEMENTE + rep)
                p = modelo.predict_proba_scores(m_aval)
                linhas.append(dict(fora=fora, modelo="T05", k=k, repeticao=rep,
                                   auc=roc_auc_score(y_aval, p),
                                   tpr5=tpr_em_fpr(y_aval, p, 0.05)))

    df = pd.DataFrame(linhas)
    SAIDA.mkdir(parents=True, exist_ok=True)
    df.to_csv(SAIDA / "item1_deixar_um_de_fora.csv", index=False)

    t05 = df[df.modelo == "T05"].groupby("k").auc.agg(["mean", "std"])
    comp = df[df.modelo != "T05"].groupby("modelo").auc.mean()
    print("Curva de diversidade da fusao (media sobre os 11 geradores deixados de fora):")
    print(f"{'k geradores no ajuste':<24}{'AUC media':>12}{'desvio':>10}")
    for k, linha in t05.iterrows():
        print(f"{k:<24.0f}{linha['mean']:>12.4f}{linha['std']:>10.4f}")
    print("\nComponentes isolados nos mesmos conjuntos de avaliacao:")
    for t, v in comp.items():
        print(f"  {t}  {v:.4f}")
    delta = t05.loc[10, "mean"] - t05.loc[1, "mean"]
    print(f"\nDe k=1 para k=10: {delta:+.4f} de AUC ({delta * 100:+.1f} pontos)")

    print("\nPor gerador deixado de fora (AUC media das 5 repeticoes):")
    tabela = (df[df.modelo == "T05"].pivot_table(index="fora", columns="k", values="auc")
              .join(df[df.modelo == "T01"].set_index("fora").auc.rename("T01 sozinha")))
    print(tabela.round(4).to_string())
    tabela.to_csv(SAIDA / "item1_por_gerador.csv")
    return df


# ----------------------------------------------------------------------
# Item 2 -- dropout de fontes
# ----------------------------------------------------------------------

def item2(dv: pd.DataFrame, ev: dict, dt: pd.DataFrame, et: dict) -> pd.DataFrame:
    print("\n" + "=" * 72)
    print("ITEM 2 -- dropout de fontes no ajuste (RN07/RNF04)")
    print("=" * 72)
    y_aj = dv.label.values
    m_aj = matriz_de(ev, dv.index.to_numpy(), len(dv))
    y_te = dt.label.values
    m_te = matriz_de(et, dt.index.to_numpy(), len(dt))

    modelos = {"padrao": ajustar(m_aj, y_aj), "dropout": ajustar(m_aj, y_aj, dropout=0.3)}

    # O vetor tem 8 colunas: 4 escores (T01-T04) + 4 bits de disponibilidade.
    # T04 nao existe nesta particao, entao a coluna 3 e a 7 saem zeradas por
    # construcao -- o que interessa sao as colunas 4 a 6.
    print("Coeficientes (T04 ausente por construcao nesta particao):")
    for nome, mod in modelos.items():
        coef = mod.classifier.coef_[0]
        print(f"  {nome:<9} escores T01-T03 {np.round(coef[:3], 3)}   "
              f"mascara T01-T03 {np.round(coef[4:7], 3)}")

    linhas = []
    cenarios = {"todas as fontes": None, **{f"sem {t}": t for t in FONTES}}
    print(f"\nAUC na particao `test` por familias ({len(dt)} imagens, 3 geradores nunca vistos):")
    print(f"{'cenario':<20}{'padrao':>10}{'dropout':>10}{'delta':>10}")
    for rotulo, ausente in cenarios.items():
        m = m_te.copy()
        if ausente is not None:
            m[:, FONTES.index(ausente)] = np.nan
        aucs = {}
        for nome, mod in modelos.items():
            p = mod.predict_proba_scores(m)
            aucs[nome] = roc_auc_score(y_te, p)
            linhas.append(dict(cenario=rotulo, modelo=nome, auc=aucs[nome],
                               tpr5=tpr_em_fpr(y_te, p, 0.05)))
        print(f"{rotulo:<20}{aucs['padrao']:>10.4f}{aucs['dropout']:>10.4f}"
              f"{aucs['dropout'] - aucs['padrao']:>+10.4f}")

    df = pd.DataFrame(linhas)
    df.to_csv(SAIDA / "item2_dropout_de_fontes.csv", index=False)
    return modelos, m_aj, y_aj, m_te, y_te


# ----------------------------------------------------------------------
# Item 2b -- o que a AUC nao ve: calibracao sob fonte ausente
# ----------------------------------------------------------------------

def item2b(modelos: dict, m_aj: np.ndarray, y_aj: np.ndarray,
           m_te: np.ndarray, y_te: np.ndarray) -> pd.DataFrame:
    """Mede deslocamento do ponto de operacao, nao ordenacao.

    A AUC nao podia mostrar diferenca aqui e isso e estrutural, nao acaso: num
    modelo linear, remover uma fonte e imputa-la por 0,5 subtrai um termo e soma
    uma constante -- a **ordenacao** das amostras restantes nao muda. O que muda
    e o valor da probabilidade, e e exatamente isso que a tela mostra ao
    usuario. Logo a medida certa e calibracao com o limiar fixo.
    """
    print("\n" + "=" * 72)
    print("ITEM 2b -- o que a AUC nao ve: o limiar se desloca quando uma fonte cai")
    print("=" * 72)

    linhas = []
    for nome, mod in modelos.items():
        # Limiar calibrado com TODAS as fontes no ar, como se faz na pratica.
        p_aj = mod.predict_proba_scores(m_aj)
        tau = float(np.quantile(p_aj[y_aj == 0], 0.95))     # FPR alvo de 5%

        for rotulo, ausente in {"todas as fontes": None,
                                **{f"sem {t}": t for t in FONTES}}.items():
            m = m_te.copy()
            if ausente is not None:
                m[:, FONTES.index(ausente)] = np.nan
            p = mod.predict_proba_scores(m)
            linhas.append(dict(
                modelo=nome, cenario=rotulo, limiar=tau,
                fpr=float((p[y_te == 0] > tau).mean()),
                tpr=float((p[y_te == 1] > tau).mean()),
                brier=float(np.mean((p - y_te) ** 2)),
                p_media_reais=float(p[y_te == 0].mean()),
            ))

    df = pd.DataFrame(linhas)
    df.to_csv(SAIDA / "item2b_calibracao.csv", index=False)

    print("Limiar fixado na `val` com todas as fontes (FPR alvo 5%), aplicado na `test`:\n")
    for metrica, rotulo, fmt in (("fpr", "FPR no limiar fixo", "{:.4f}"),
                                 ("tpr", "deteccao no limiar fixo", "{:.4f}"),
                                 ("brier", "Brier (menor e melhor)", "{:.4f}")):
        tabela = df.pivot(index="cenario", columns="modelo", values=metrica)
        tabela = tabela.reindex(["todas as fontes", "sem T01", "sem T02", "sem T03"])
        print(f"### {rotulo}")
        print(tabela.applymap(fmt.format).to_string(), "\n")

    base = df[(df.modelo == "padrao") & (df.cenario == "todas as fontes")].fpr.iloc[0]
    print(f"FPR de referencia (padrao, todas as fontes): {base:.4f}")
    for nome in modelos:
        pior = df[(df.modelo == nome) & (df.cenario != "todas as fontes")].fpr
        print(f"  {nome:<9} FPR sob fonte ausente: min {pior.min():.4f}  max {pior.max():.4f}")
    return df


if __name__ == "__main__":
    dv, ev, dt, et = passo0()
    item1(dv, ev)
    modelos, m_aj, y_aj, m_te, y_te = item2(dv, ev, dt, et)
    item2b(modelos, m_aj, y_aj, m_te, y_te)
    print(f"\nCSVs em {SAIDA.relative_to(RAIZ)}/")
