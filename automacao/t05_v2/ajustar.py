"""Passo 3 - ajusta e compara as variantes da T05 para a tela v2.

Quatro variantes, o cruzamento de duas decisoes em aberto:

    fontes    {T01, T02, T04}          a T03 fora
              {T01, T02, T03, T04}     a T03 dentro
    mascara   disponibilidade          o que o TCC 2 descreve: respondeu / nao
              confiabilidade           acrescenta "respondeu com entrada
                                       degenerada"

A T01 e a **reimplementacao** (o modelo que a tela v2 carrega), pontuada em
`resultados/t05_v2/t01_replica_*`. As outras tres vem do cache de 256 px.

Mascara de confiabilidade
-------------------------
Hoje a T05 recebe 4 bits que dizem "a fonte respondeu". Isso nao distingue um
escore util de um escore que a propria fonte sabe ser lixo. O caso medido: o
classificador objeto-sombra da T04 devolve a **constante** 0,350257 quando o
SSISv2 nao acha nenhum par -- 44,5% das imagens da particao `test`. A Secao 4.7
de RESULTADOS.md ja mede AUC 0,4946 nesse subconjunto (o acaso) e mostra que
essas linhas **rebaixam** a AUC agregada em 8 pontos.

⚠️ **Este bit tambem e um atalho em potencial, e a direcao dele nao e estavel:**

    amostragem inicial (t04_mascaras_vazias.py)  reais 45,0%  sinteticas 24,7%
    particao fusion                              reais 31,9%  sinteticas 65,3%
    particao test                                reais 30,7%  sinteticas 48,4%

A amostragem dizia "vazia -> real"; no OOD e "vazia -> sintetica". Se a variante
de confiabilidade ganhar, e **obrigatorio** verificar se ela ganhou por
descontar ruido ou por explorar essa correlacao -- que inverte entre corpora e
nao sobreviveria a um gerador novo. O relatorio imprime as taxas para que a
pergunta nao passe batida.
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

RAIZ = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(RAIZ))

from codigo.metricas import bootstrap_auc_ci, compute_metrics, delong_test  # noqa: E402

SAIDA = RAIZ / "resultados" / "t05_v2"
CONSTANTE_MASCARA_VAZIA = 0.350257
TOLERANCIA = 1e-5

FONTES_COM_T03 = ("T01", "T02", "T03", "T04")
FONTES_SEM_T03 = ("T01", "T02", "T04")


def chave(valor: str) -> str:
    partes = str(valor).replace("\\", "/").rstrip("/").split("/")
    return "/".join(partes[-2:]).rsplit(".", 1)[0].lower()


def radical(valor: str) -> str:
    """So o nome do arquivo, sem pasta nem extensao.

    Os CSV de componentes da T04 gravam ``arquivo`` como radical puro
    (``coco_000111``), sem a pasta do gerador -- diferente dos manifestos. Como
    todo nome do corpus e prefixado pelo gerador, o radical identifica a imagem
    sem ambiguidade, e e por ele que a T04 e juntada.
    """
    return str(valor).replace("\\", "/").rstrip("/").split("/")[-1].rsplit(".", 1)[0].lower()


def montar(particao: str) -> pd.DataFrame:
    """Uma linha por imagem, com as quatro fontes e os sinalizadores."""
    indice = pd.read_csv(SAIDA / "indice_limpeza.csv")
    base = indice[indice["split"] == particao].copy()
    base["k"] = base["path"].map(chave)

    # --- T01: a replica ------------------------------------------------------
    pasta = "t01_replica_fusion" if particao == "fusion" else "t01_replica_test"
    t01 = pd.read_csv(SAIDA / pasta / "escores.csv")
    t01["k"] = t01["path"].map(chave)
    base = base.merge(
        t01[["k", "escore"]].rename(columns={"escore": "T01"}),
        on="k", how="left", validate="one_to_one",
    )

    # --- T02 e T03 -----------------------------------------------------------
    if particao == "fusion":
        ordem = pd.read_csv(RAIZ / "data" / "manifesto30k_ood.csv")
        ordem = ordem[ordem["split"] == "fusion"].reset_index(drop=True)
        ordem["k"] = ordem["path"].map(chave)
        for nome in ("T02", "T03"):
            vetor = np.load(
                RAIZ / "resultados" / "scores" / f"ood__{nome}__{nome}__fusion.npy"
            )
            if len(vetor) != len(ordem):
                raise SystemExit(f"cache de {nome} desalinhado da particao fusion")
            base = base.merge(
                pd.DataFrame({"k": ordem["k"], nome: vetor}),
                on="k", how="left", validate="one_to_one",
            )
    else:
        for nome in ("T02", "T03"):
            arquivo = SAIDA / f"{nome.lower()}_256_test.csv"
            if not arquivo.exists():
                raise SystemExit(
                    f"{arquivo.name} ausente -- o passo 2 ainda nao terminou"
                )
            bruto = pd.read_csv(arquivo)
            bruto["k"] = bruto["path"].map(chave)
            base = base.merge(
                bruto[["k", "escore"]].rename(columns={"escore": nome}),
                on="k", how="left", validate="one_to_one",
            )

    # --- T04: media das tres representacoes, mais o sinalizador --------------
    sufixo = "_ood_fusion" if particao == "fusion" else "_ood"
    comp = pd.read_csv(RAIZ / "resultados" / f"t04_escores_componentes{sufixo}.csv")
    comp["r"] = comp["arquivo"].map(radical)
    representacoes = ["object_shadow", "perspective_fields", "line_segment"]
    comp["T04"] = comp[representacoes].mean(axis=1, skipna=True)
    comp["os_vazia"] = (
        comp["object_shadow"].sub(CONSTANTE_MASCARA_VAZIA).abs() < TOLERANCIA
    ).astype(int)
    comp = comp.drop_duplicates(subset="r")

    base["r"] = base["path"].map(radical)
    juntado = base.merge(
        comp[["r", "T04", "os_vazia"]], on="r", how="left", validate="one_to_one"
    )
    ausentes = int(juntado["T04"].isna().sum())
    if ausentes:
        # Nao e fatal -- a imputacao cobre --, mas silenciar seria repetir o
        # erro que produziu a primeira rodada com a T04 inteira em NaN.
        print(f"   ⚠️ T04 ausente em {ausentes}/{len(juntado)} linhas de {particao}")
    return juntado


def vetorizar(quadro: pd.DataFrame, fontes, confiabilidade: bool) -> np.ndarray:
    escores = quadro[list(fontes)].to_numpy(dtype=np.float64)
    disponivel = (~np.isnan(escores)).astype(np.float64)
    preenchido = np.where(np.isnan(escores), 0.5, escores)
    partes = [preenchido, disponivel]
    if confiabilidade:
        # Um bit por fonte: 1 = respondeu com entrada utilizavel. Hoje so a T04
        # sabe declarar isso; as demais repetem a disponibilidade.
        util = disponivel.copy()
        if "T04" in fontes:
            coluna = list(fontes).index("T04")
            util[:, coluna] = disponivel[:, coluna] * (
                1 - quadro["os_vazia"].fillna(0).to_numpy()
            )
        partes.append(util)
    return np.hstack(partes)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--so-ajustar", action="store_true",
                        help="nao avalia; util enquanto o passo 2 roda")
    args = parser.parse_args()

    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    ajuste = montar("fusion")
    limpo = ajuste[ajuste["limpa"]].copy()
    print(f"ajuste: {len(ajuste)} linhas, {len(limpo)} limpas para a replica")
    print(f"   rotulos: {dict(limpo.label.value_counts())}")
    print("\nescores medios por fonte e rotulo:")
    print(limpo.groupby("label")[["T01", "T02", "T03", "T04"]].mean().to_string())
    print("\ntaxa de mascara objeto-sombra vazia, por rotulo:")
    print(limpo.groupby("label")["os_vazia"].mean().to_string())

    avaliacao = None
    if not args.so_ajustar:
        avaliacao = montar("test")
        avaliacao = avaliacao[avaliacao["limpa"]].copy()
        print(f"\navaliacao: {len(avaliacao)} linhas limpas")

    resultados: dict[str, dict] = {}
    for rotulo_fontes, fontes in (("sem_T03", FONTES_SEM_T03),
                                  ("com_T03", FONTES_COM_T03)):
        for rotulo_mascara, confiavel in (("disponibilidade", False),
                                          ("confiabilidade", True)):
            nome = f"{rotulo_fontes}__{rotulo_mascara}"
            X = vetorizar(limpo, fontes, confiavel)
            y = limpo["label"].to_numpy()
            escalador = StandardScaler().fit(X)
            modelo = LogisticRegression(max_iter=1000, random_state=42)
            modelo.fit(escalador.transform(X), y)

            pesos = dict(zip(fontes, modelo.coef_[0][: len(fontes)]))
            registro = {
                "fontes": list(fontes),
                "mascara": rotulo_mascara,
                "pesos": {k: round(float(v), 4) for k, v in pesos.items()},
            }

            if avaliacao is not None:
                Xa = vetorizar(avaliacao, fontes, confiavel)
                p = modelo.predict_proba(escalador.transform(Xa))[:, 1]
                ya = avaliacao["label"].to_numpy()
                registro["metricas"] = {
                    k: round(float(v), 4)
                    for k, v in compute_metrics(ya, p).items()
                    if isinstance(v, (int, float))
                }
                baixo, alto = bootstrap_auc_ci(ya, p, n_resamples=1000)
                registro["ic95"] = [round(baixo, 4), round(alto, 4)]
                avaliacao[f"p__{nome}"] = p

            with (SAIDA / f"t05_v2__{nome}.pkl").open("wb") as arquivo:
                pickle.dump(
                    {"classifier": modelo, "scaler": escalador,
                     "sources": list(fontes), "mascara": rotulo_mascara},
                    arquivo,
                )
            resultados[nome] = registro

    print("\n" + "=" * 72)
    for nome, r in resultados.items():
        print(f"\n{nome}")
        print("   pesos:", r["pesos"])
        if "metricas" in r:
            m = r["metricas"]
            print(f"   AUC {m.get('auc')}  IC {r['ic95']}  "
                  f"acc {m.get('accuracy')}  FPR {m.get('fpr')}")

    if avaliacao is not None:
        print("\n--- DeLong entre variantes, no mesmo conjunto ---")
        y = avaliacao["label"].to_numpy()
        nomes = list(resultados)
        for i in range(len(nomes)):
            for j in range(i + 1, len(nomes)):
                a = avaliacao[f"p__{nomes[i]}"].to_numpy()
                b = avaliacao[f"p__{nomes[j]}"].to_numpy()
                t = delong_test(y, a, b)
                print(f"   {nomes[i]} x {nomes[j]}: "
                      f"dif {t['difference']:+.4f}  p {t['p_value']:.4g}")
        avaliacao.to_csv(SAIDA / "avaliacao_pareada.csv", index=False)

    (SAIDA / "variantes.json").write_text(
        json.dumps(resultados, indent=2), encoding="utf-8"
    )
    print(f"\ngravado em {SAIDA}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
