"""A alavanca da diversidade aplicada a T03 -- a unica ainda nao testada nela.

Por que so a T03
----------------
Das quatro tecnicas-fonte, treinar so muda o que se pode mudar:

* **T02** usa pesos oficiais do SPAI e nao e treinavel neste trabalho. Nao ha
  ajuste a fazer -- so integrar e calibrar limiar.
* **T04** tem classificadores oficiais mas os extratores (PerspectiveFields,
  SSISv2, ambos com detectron2) nao acompanham; esta bloqueada por fora do
  repositorio, e o p = 0,182 diz que nao paga desbloquear.
* **T01** ja recebeu a alavanca -- e dela que vem o retreino de 05/09.
* **T05** ja foi reajustada em 05/09; o que resta nela sao decisoes, nao codigo.

Sobra a **T03**, e nela a busca em grade ja esta embutida no `fit()`
(`GridSearchCV` sobre `n_estimators` x `max_depth`, 5 particoes) -- ou seja,
hiperparametro ja esta no teto. O que **nunca** foi testado e a diversidade de
geradores sob volume constante, que na T01 valeu +14,1 pontos.

O desenho, e o controle que o torna interpretavel
-------------------------------------------------
Dois bracos com **exatamente o mesmo numero de imagens**, mudando so a variedade:

* `balanceado_11` -- 850 sinteticas de cada um dos 11 geradores da `train`
* `unico_ld`      -- 9.350 sinteticas so de `latent_diffusion`

Sem o segundo braco, um ganho poderia vir do volume ou do recorte, e nao da
diversidade. Com ele, a diferenca entre os dois **e** o efeito da diversidade --
mesmo controle do Achado 14 da replicacao.

A T03 entregue por familias (`pesos_ood_familias/`, AUC 0,6956 na `test`) entra
como terceira referencia, ja publicada, mas ⚠️ ela usou 55.000 imagens: nao e
comparavel em volume com nenhum dos dois bracos, so em ordem de grandeza.

Passo 2 -- a fusao herda o que a T03 ganhar
-------------------------------------------
Se a T03 melhorar, os escores dela mudam e a T05 precisa ser reajustada sobre os
novos. O script faz isso e reporta o antes/depois -- sem tocar em `pesos/`.

Custo medido: extracao a ~58 ms/imagem (medido em 05/09 sobre 20 imagens), mais
duas buscas em grade. Estimativa de ~2 h. As caracteristicas sao gravadas em
`resultados/t03_diversidade/features/` para que uma reexecucao nao repita a
parte cara.

Uso::

    python automacao/t03_diversidade.py
    python automacao/t03_diversidade.py --so-guarda   # confere alinhamento e sai
"""

from __future__ import annotations

import argparse
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ))

from codigo.tecnicas.t03_benford.tecnica import T03Benford  # noqa: E402
from codigo.tecnicas.t05_fusao.tecnica import T05Fusion, build_score_matrix  # noqa: E402

MANIFESTO = RAIZ / "data" / "manifesto30k_ood_familias.csv"
CACHE = RAIZ / "resultados" / "scores"
SAIDA = RAIZ / "resultados" / "t03_diversidade"
FEATURES = SAIDA / "features"
COFRE = Path.home() / "OneDrive" / "Documentos" / "TCC 3"

POR_GERADOR = 850               # o teto dos dez geradores menores da `train`
SEMENTE = 42

# Secao 3.4 de documentacao/RESULTADOS.md -- guarda e referencia publicada.
AUC_PUBLICADA_TEST = {"T01": 0.9463, "T02": 0.6794, "T03": 0.6956}
AUC_T05_PUBLICADA = 0.9261


def particao(split: str) -> pd.DataFrame:
    d = pd.read_csv(MANIFESTO)
    return d[d.split == split].reset_index(drop=True)


def escores_cache(split_arquivo: str, n: int, tecnicas=("T01", "T02")) -> dict[str, np.ndarray]:
    saida = {}
    for t in tecnicas:
        a = np.load(CACHE / f"ood__{t}__{t}__{split_arquivo}.npy")
        if a.shape != (n,):
            raise SystemExit(f"desalinhamento em {t}/{split_arquivo}: {a.shape[0]} x {n}")
        saida[t] = a
    return saida


def caracteristicas(rotulo: str, caminhos: list[Path]) -> np.ndarray:
    """Extrai (ou reaproveita) as 540 dimensoes de Benford.

    E a etapa cara do pipeline -- a unica que toca imagem. Gravar em disco faz
    a diferenca entre reexecutar em minutos e reexecutar em horas.
    """
    arquivo = FEATURES / f"{rotulo}.npy"
    if arquivo.exists():
        X = np.load(arquivo)
        if len(X) == len(caminhos):
            print(f"[features] {rotulo}: reaproveitadas ({X.shape})", flush=True)
            return X
    FEATURES.mkdir(parents=True, exist_ok=True)
    inicio = time.time()
    X = T03Benford().extract_features(caminhos)
    np.save(arquivo, X)
    print(f"[features] {rotulo}: {X.shape} em {(time.time() - inicio) / 60:.1f} min", flush=True)
    return X


def montar_bracos(treino: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Dois recortes de mesmo tamanho: um diverso, um de gerador unico."""
    aleatorio = np.random.default_rng(SEMENTE)
    geradores = sorted(g for g in treino.generator.unique() if g != "real")

    def sortear(d: pd.DataFrame, n: int) -> pd.DataFrame:
        if len(d) < n:
            raise SystemExit(f"ERRO: preciso de {n} linhas, ha {len(d)}")
        return d.iloc[aleatorio.choice(len(d), n, replace=False)]

    diversas = pd.concat([sortear(treino[treino.generator == g], POR_GERADOR)
                          for g in geradores])
    n_sinteticas = len(diversas)
    unicas = sortear(treino[treino.generator == "latent_diffusion"], n_sinteticas)
    reais = sortear(treino[treino.label == 0], n_sinteticas)

    return {
        "balanceado_11": pd.concat([diversas, reais]).reset_index(drop=True),
        "unico_ld": pd.concat([unicas, reais]).reset_index(drop=True),
    }


def tabela_md(d: pd.DataFrame, casas: int = 4) -> str:
    def fmt(v) -> str:
        return f"{v:.{casas}f}" if isinstance(v, (float, np.floating)) else str(v)
    return "\n".join([
        "| " + " | ".join(d.columns) + " |",
        "|" + "|".join("---" for _ in d.columns) + "|",
        *["| " + " | ".join(fmt(v) for v in linha) + " |" for linha in d.itertuples(index=False)],
    ])


def executar(so_guarda: bool) -> list[str]:
    linhas: list[str] = []
    treino, val, teste = particao("train"), particao("val"), particao("test")

    # --- guarda de alinhamento -------------------------------------------
    et = escores_cache("clean", len(teste), ("T01", "T02", "T03"))
    for t, alvo in AUC_PUBLICADA_TEST.items():
        medido = roc_auc_score(teste.label.values, et[t])
        if abs(medido - alvo) > 5e-4:
            return [f"⛔ **Guarda reprovada** — {t} mede {medido:.4f}, publicado {alvo:.4f}. "
                    "Nada foi treinado."]
    linhas += ["Guarda de alinhamento: as três AUCs da Seção 3.4 reproduzidas do cache. "
               f"Referência da T03 a bater: **{AUC_PUBLICADA_TEST['T03']:.4f}** "
               "(treinada com 55.000 imagens, 11 geradores desbalanceados).", ""]
    if so_guarda:
        return linhas

    # --- caracteristicas --------------------------------------------------
    bracos = montar_bracos(treino)
    X_teste = caracteristicas("teste", [Path(p) for p in teste.path])
    X_val = caracteristicas("val", [Path(p) for p in val.path])

    registros = []
    escores_novos = {}
    for nome, recorte in bracos.items():
        n_ger = recorte[recorte.label == 1].generator.nunique()
        print(f"\n===== {nome}: {len(recorte)} imagens, {n_ger} geradores =====", flush=True)
        X = caracteristicas(nome, [Path(p) for p in recorte.path])

        inicio = time.time()
        modelo = T03Benford(seed=SEMENTE).fit(list(recorte.path), recorte.label.values, features=X)
        minutos = (time.time() - inicio) / 60

        p_teste = modelo.predict_proba(list(teste.path), features=X_teste)
        p_val = modelo.predict_proba(list(val.path), features=X_val)
        escores_novos[nome] = (p_val, p_teste)
        auc = roc_auc_score(teste.label.values, p_teste)
        print(f"  -> AUC na test = {auc:.4f} ({minutos:.0f} min)", flush=True)
        registros.append(dict(braco=nome, imagens=len(recorte), geradores=n_ger,
                              auc_test=auc, minutos=minutos,
                              **{k: v for k, v in modelo.best_params_.items()
                                 if k in ("n_estimators", "max_depth", "cv_auc")}))

    d = pd.DataFrame(registros)
    SAIDA.mkdir(parents=True, exist_ok=True)
    d.to_csv(SAIDA / "bracos.csv", index=False)
    linhas += ["## Passo 1 — a T03 sob diversidade", "", tabela_md(d), ""]

    div = float(d[d.braco == "balanceado_11"].auc_test.iloc[0])
    uni = float(d[d.braco == "unico_ld"].auc_test.iloc[0])
    delta = div - uni
    linhas += [f"**Efeito da diversidade, a volume constante: {delta:+.4f} de AUC "
               f"({delta * 100:+.1f} pontos).**", ""]
    if delta > 0.02:
        linhas += ["🟢 A alavanca que funcionou na T01 **também funciona na T03**. "
                   "Vale reportar como resultado de método, não só de técnica: o defeito "
                   "é do protocolo de treino com uma classe sintética, e atravessa "
                   "famílias de técnica diferentes.", ""]
    elif delta > 0.005:
        linhas += ["🟠 Efeito presente mas pequeno — bem abaixo dos +14,1 pontos da T01. "
                   "Reportar como contraste: a alavanca não é universal.", ""]
    else:
        linhas += ["🔴 Sem efeito. A T03 não se conserta por diversidade, e isso **fortalece** "
                   "a leitura da T01: lá o ganho vem de a coocorrência memorizar a assinatura "
                   "de um gerador, um mecanismo que a estatística de Benford não tem.", ""]

    # --- passo 2: a fusao herda -------------------------------------------
    ev = escores_cache("val", len(val))
    melhor = "balanceado_11" if div >= uni else "unico_ld"
    p_val, p_teste = escores_novos[melhor]

    antes = T05Fusion(seed=SEMENTE).fit_scores(
        build_score_matrix({**ev, "T03": np.load(CACHE / "ood__T03__T03__val.npy")}, len(val)),
        val.label.values)
    depois = T05Fusion(seed=SEMENTE).fit_scores(
        build_score_matrix({**ev, "T03": p_val}, len(val)), val.label.values)

    auc_antes = roc_auc_score(teste.label.values, antes.predict_proba_scores(
        build_score_matrix(et, len(teste))))
    auc_depois = roc_auc_score(teste.label.values, depois.predict_proba_scores(
        build_score_matrix({"T01": et["T01"], "T02": et["T02"], "T03": p_teste}, len(teste))))

    linhas += ["## Passo 2 — a fusão sobre a T03 nova", "",
               f"Braço adotado: **{melhor}**.", "",
               f"| T05 | AUC na `test` |", "|---|---|",
               f"| com a T03 atual | {auc_antes:.4f} |",
               f"| com a T03 nova | {auc_depois:.4f} |", "",
               f"Δ = **{auc_depois - auc_antes:+.4f}**. ⚠️ A referência publicada da T05 é "
               f"{AUC_T05_PUBLICADA:.4f}, e a T01 sozinha faz {AUC_PUBLICADA_TEST['T01']:.4f} "
               "nesta mesma partição — **nenhum ganho aqui reabre a manchete de AUC** "
               "enquanto a fusão não passar da T01. Ver [[T05 — Fusão tardia]].", ""]
    return linhas


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--so-guarda", action="store_true")
    parser.add_argument("--sem-cofre", action="store_true")
    args = parser.parse_args()

    for fluxo in (sys.stdout, sys.stderr):
        try:
            fluxo.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass

    agora = datetime.now()
    corpo = ["---", "tipo: resultados", "tags: [tcc3, resultados, experimento]",
             f"atualizado: {agora:%Y-%m-%d}", "---", "",
             "# T03 sob diversidade de geradores", "",
             f"> Gerado por `automacao/t03_diversidade.py` em {agora:%d/%m/%Y às %H:%M}. "
             "Ver [[Plano pós-retreino]].", ""]
    try:
        corpo += executar(args.so_guarda)
    except Exception:                                       # noqa: BLE001
        corpo += ["⛔ **A execução quebrou.**", "", "```"] + \
                 traceback.format_exc().splitlines() + ["```", ""]

    texto = "\n".join(corpo)
    SAIDA.mkdir(parents=True, exist_ok=True)
    (SAIDA / "relatorio.md").write_text(texto, encoding="utf-8")
    if not args.sem_cofre and COFRE.exists():
        (COFRE / "Resultados" / "T03 sob diversidade.md").write_text(texto, encoding="utf-8")
    print(texto)
    return 0


class _Espelho:
    """Escreve nos dois destinos: o arquivo de log e o console, se houver."""

    def __init__(self, arquivo, console):
        self._destinos = [d for d in (arquivo, console) if d is not None]

    def write(self, texto: str) -> int:
        for destino in self._destinos:
            try:
                destino.write(texto)
            except Exception:       # noqa: BLE001  console morto nao mata o log
                pass
        return len(texto)

    def flush(self) -> None:
        for destino in self._destinos:
            try:
                destino.flush()
            except Exception:       # noqa: BLE001
                pass

    def isatty(self) -> bool:
        return False


def _abrir_log():
    """Abre o log SEMPRE e espelha no console quando ele existe.

    Por que incondicional (06/09/2026)
    ----------------------------------
    A versao anterior so redirecionava se ``sys.stdout is None or not
    sys.stdout.isatty()``. Sob o Agendador essa guarda **nao disparou**: a
    execucao das 22h de 06/09 rodou por 26 min sem escrever uma linha, e o
    progresso se perdeu inteiro. E a armadilha ja conhecida ("tarefa do
    Agendador nao tem console") voltando pela porta da guarda condicional.

    Os drivers que nunca falharam (`noite_retreino.py`, `noite_raise1k.py`)
    abrem o log sem condicao — e e o que se faz aqui. O espelho preserva a
    saida no terminal para quem roda o script a mao, que era a unica coisa que
    a guarda protegia.
    """
    caminho = RAIZ / "logs" / "t03_diversidade.log"
    caminho.parent.mkdir(parents=True, exist_ok=True)
    arquivo = open(caminho, "a", encoding="utf-8", errors="replace", buffering=1)

    console = None
    fluxo = sys.__stdout__
    if fluxo is not None:
        try:
            if fluxo.isatty():
                console = fluxo
        except Exception:           # noqa: BLE001  handle invalido sob o Agendador
            console = None

    sys.stdout = sys.stderr = _Espelho(arquivo, console)
    return arquivo


if __name__ == "__main__":
    _abrir_log()
    sys.exit(main())
