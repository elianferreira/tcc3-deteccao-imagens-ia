"""AUC das tecnicas trocando a classe real: COCO x RAISE-1k.

Continuacao de `fpr_raise1k_todas.py`, que mediu so a FPR. Aqui entra o lado
sintetico, e com ele a AUC — a metrica que o Capitulo 4 reporta.

Por que reescorear os sinteticos em vez de usar o cache
------------------------------------------------------
Ha escores em `resultados/scores/ood__*.npy`, mas os tamanhos nao batem de
imediato com nenhum split obvio (`ood__T03__T03__clean` tem 7.138 linhas; o
`test` do `manifesto_escala_ood` tem 21.088). Casar isso exigiria reconstruir a
correspondencia por fora — e desalinhamento de escore e a familia de armadilha
que mais reincide neste projeto.

Aqui o script escoreia os dois lados **no mesmo processo, na mesma ordem**, com
o mesmo codigo que reproduziu a FPR publicada na terceira casa decimal. Custa
~50 min de T02 e elimina a duvida.

Os escores REAIS sao reaproveitados de `resultados/fpr_raise1k/` (ja medidos as
10h50), porque foram produzidos por este mesmo caminho.

O cache de sinteticos e gravado: reexecutar so recalcula a AUC.

Uso::

    python automacao/auc_raise1k_todas.py
    python automacao/auc_raise1k_todas.py --n 300     # ensaio
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

RAIZ = Path(__file__).resolve().parents[1]
if str(RAIZ) not in sys.path:
    sys.path.insert(0, str(RAIZ))

from codigo.configuracao import DECISION_THRESHOLD, WEIGHTS_DIR  # noqa: E402

MANIFESTO_OOD = RAIZ.parent / "tcc3-t01-replicacao" / "data" / "manifesto_ood13.csv"
ORIGEM_REAIS = RAIZ / "resultados" / "fpr_raise1k"
SAIDA = RAIZ / "resultados" / "auc_raise1k"
COFRE = Path(r"C:\Users\ferre\OneDrive\Documentos\TCC 3")

TECNICAS = ("T01", "T02", "T03", "T05")


def auc(rotulos: np.ndarray, valores: np.ndarray) -> float:
    rotulos = np.asarray(rotulos)
    valores = np.asarray(valores, dtype=float)
    if len(np.unique(rotulos)) < 2:
        return float("nan")
    ordem = np.argsort(valores, kind="mergesort")
    v, y = valores[ordem], rotulos[ordem]
    postos = np.empty(len(v))
    i = 0
    while i < len(v):
        j = i
        while j + 1 < len(v) and v[j + 1] == v[i]:
            j += 1
        postos[i:j + 1] = (i + j) / 2.0 + 1.0
        i = j + 1
    npos = int((y == 1).sum())
    nneg = len(y) - npos
    return float((postos[y == 1].sum() - npos * (npos + 1) / 2.0) / (npos * nneg))


def ic_auc(reais: np.ndarray, sinteticas: np.ndarray,
           reamostragens: int = 1000, semente: int = 42) -> tuple[float, float]:
    rng = np.random.default_rng(semente)
    amostras = []
    for _ in range(reamostragens):
        r = reais[rng.integers(0, len(reais), len(reais))]
        s = sinteticas[rng.integers(0, len(sinteticas), len(sinteticas))]
        rot = np.r_[np.zeros(len(r), int), np.ones(len(s), int)]
        amostras.append(auc(rot, np.r_[r, s]))
    return (float(np.percentile(amostras, 2.5)),
            float(np.percentile(amostras, 97.5)))


def sinteticas_do_benchmark(n: int | None) -> tuple[list[Path], list[str]]:
    caminhos, geradores = [], []
    with open(MANIFESTO_OOD, newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            if int(r["label"]) == 1:
                p = RAIZ / r["path"].replace("\\", "/")
                if p.exists():
                    caminhos.append(p)
                    geradores.append(r["generator"])
    if n:
        caminhos, geradores = caminhos[:n], geradores[:n]
    return caminhos, geradores


def escorear(caminhos: list[Path]) -> dict[str, np.ndarray]:
    """Mesmo caminho de `fpr_raise1k_todas.py`, com cache em disco."""
    escores: dict[str, np.ndarray] = {}

    def cache(tecnica: str) -> Path:
        return SAIDA / f"sinteticas__{tecnica}.npy"

    from codigo.tecnicas.t01_coocorrencia import T01Cooccurrence
    if cache("T01").exists() and len(np.load(cache("T01"))) == len(caminhos):
        escores["T01"] = np.load(cache("T01"))
        print("  T01: cache", flush=True)
    else:
        t = time.time()
        m = T01Cooccurrence(device="cuda")
        m.load(WEIGHTS_DIR / "t01_cooccurrence.pt")
        p = np.asarray(m.predict_proba(caminhos), dtype=np.float64)
        escores["T01"] = p[:, 1] if p.ndim > 1 else p
        np.save(cache("T01"), escores["T01"])
        print(f"  T01 em {(time.time()-t)/60:.1f} min", flush=True)

    from codigo.tecnicas.t03_benford import T03Benford
    if cache("T03").exists() and len(np.load(cache("T03"))) == len(caminhos):
        escores["T03"] = np.load(cache("T03"))
        print("  T03: cache", flush=True)
    else:
        t = time.time()
        m = T03Benford(n_jobs=4).load(WEIGHTS_DIR / "t03_benford.pkl")
        p = np.asarray(m.predict_proba(caminhos), dtype=np.float64)
        escores["T03"] = p[:, 1] if p.ndim > 1 else p
        np.save(cache("T03"), escores["T03"])
        print(f"  T03 em {(time.time()-t)/60:.1f} min", flush=True)

    from codigo.tecnicas.t02_spai.tecnica import T02SPAI
    if cache("T02").exists() and len(np.load(cache("T02"))) == len(caminhos):
        escores["T02"] = np.load(cache("T02"))
        print("  T02: cache", flush=True)
    else:
        t = time.time()
        m = T02SPAI()
        ok, motivo = m.is_available()
        if not ok:
            print(f"  T02 INDISPONIVEL: {motivo}", flush=True)
        else:
            p = np.asarray(m.predict_proba(caminhos), dtype=np.float64)
            escores["T02"] = p[:, 1] if p.ndim > 1 else p
            np.save(cache("T02"), escores["T02"])
            print(f"  T02 em {(time.time()-t)/60:.1f} min", flush=True)

    from codigo.tecnicas.t05_fusao.tecnica import T05Fusion, build_score_matrix
    try:
        t05 = T05Fusion().load(WEIGHTS_DIR / "t05_fusion.pkl")
        p = np.asarray(
            t05.predict_proba_scores(build_score_matrix(escores, len(caminhos))),
            dtype=np.float64)
        escores["T05"] = p[:, 1] if p.ndim > 1 else p
    except Exception as erro:                              # noqa: BLE001
        print(f"  T05 FALHOU: {erro}", flush=True)

    return escores


def carregar_reais() -> dict[str, dict[str, np.ndarray]]:
    reais: dict[str, dict[str, np.ndarray]] = {"coco": {}, "raise1k": {}}
    for amostra in reais:
        for tecnica in TECNICAS:
            caminho = ORIGEM_REAIS / f"escores__{amostra}__{tecnica}.npy"
            if caminho.exists():
                reais[amostra][tecnica] = np.load(caminho)
    return reais


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=None)
    parser.add_argument("--sem-cofre", action="store_true")
    args = parser.parse_args(argv)

    SAIDA.mkdir(parents=True, exist_ok=True)
    print(f"=== inicio {datetime.now():%Y-%m-%d %H:%M:%S} ===", flush=True)

    caminhos, geradores = sinteticas_do_benchmark(args.n)
    geradores = np.asarray(geradores)
    print(f"sinteticas: {len(caminhos)} de "
          f"{len(set(geradores.tolist()))} geradores\n", flush=True)

    sint = escorear(caminhos)
    reais = carregar_reais()
    print(f"\nreais reaproveitadas: coco={len(reais['coco'])} tecnicas, "
          f"raise1k={len(reais['raise1k'])} tecnicas", flush=True)

    resultado: dict = {"n_sinteticas": len(caminhos), "por_tecnica": {}}
    linhas = [
        "---", "tipo: resultados", "tags: [tcc3, resultados, corpus]",
        f"atualizado: {datetime.now():%Y-%m-%d}", "---", "",
        "# AUC trocando a classe real: COCO x RAISE-1k", "",
        f"> Gerado por `automacao/auc_raise1k_todas.py` em "
        f"{datetime.now():%d/%m/%Y as %H:%M}. {len(caminhos)} sinteticas dos 13 "
        f"geradores contra 999 reais de cada fonte.",
        "> **T04 como NaN nas duas colunas** — a T05 aqui e de tres fontes.", "",
        "| tecnica | AUC com COCO | AUC com RAISE-1k | queda |",
        "|---|---|---|---|",
    ]

    for tecnica in TECNICAS:
        if tecnica not in sint:
            continue
        s = sint[tecnica]
        linha = {"n_sinteticas": len(s)}
        celulas = []
        for amostra in ("coco", "raise1k"):
            r = reais[amostra].get(tecnica)
            if r is None:
                celulas.append("—")
                continue
            rot = np.r_[np.zeros(len(r), int), np.ones(len(s), int)]
            a = auc(rot, np.r_[r, s])
            lo, hi = ic_auc(r, s)
            linha[amostra] = {"auc": a, "ic_baixo": lo, "ic_alto": hi}
            celulas.append(f"{a:.4f} [{lo:.3f}; {hi:.3f}]")
        if "coco" in linha and "raise1k" in linha:
            d = linha["raise1k"]["auc"] - linha["coco"]["auc"]
            linha["delta"] = d
            celulas.append(f"**{d:+.4f}**")
        else:
            celulas.append("—")
        resultado["por_tecnica"][tecnica] = linha
        linhas.append(f"| **{tecnica}** | {celulas[0]} | {celulas[1]} | {celulas[2]} |")

    # por gerador, so para a T05 e a T02 (a que resiste e a proposta)
    linhas += ["", "## Por gerador, com o RAISE como classe real", "",
               "| gerador | " + " | ".join(TECNICAS) + " |",
               "|---" * (len(TECNICAS) + 1) + "|"]
    for g in sorted(set(geradores.tolist())):
        m = geradores == g
        celulas = []
        for tecnica in TECNICAS:
            r = reais["raise1k"].get(tecnica)
            if tecnica not in sint or r is None:
                celulas.append("—")
                continue
            s = sint[tecnica][m]
            rot = np.r_[np.zeros(len(r), int), np.ones(len(s), int)]
            celulas.append(f"{auc(rot, np.r_[r, s]):.4f}")
        linhas.append(f"| {g} | " + " | ".join(celulas) + " |")

    linhas += ["", "⚠️ **O script so mede** — a leitura de sentido entra na nota "
               "de sessao do dia.", ""]

    texto = "\n".join(linhas)
    (SAIDA / "resumo.json").write_text(
        json.dumps(resultado, ensure_ascii=False, indent=2, default=float),
        encoding="utf-8")
    (SAIDA / "relatorio.md").write_text(texto, encoding="utf-8")
    if not args.sem_cofre and COFRE.exists():
        (COFRE / "Resultados" / "AUC com o RAISE como classe real.md").write_text(
            texto, encoding="utf-8")
    print("\n" + texto, flush=True)
    print(f"=== fim {datetime.now():%Y-%m-%d %H:%M:%S} ===", flush=True)
    return 0


def _abrir_log():
    caminho = RAIZ / "logs" / "auc_raise1k_todas.log"
    caminho.parent.mkdir(parents=True, exist_ok=True)
    return open(caminho, "a", encoding="utf-8", errors="replace", buffering=1)


if __name__ == "__main__":
    arquivo = _abrir_log()
    console = sys.__stdout__
    try:
        if console is not None and not console.isatty():
            console = None
    except Exception:                                       # noqa: BLE001
        console = None

    class _Espelho:
        def write(self, t):
            for d in (arquivo, console):
                if d is not None:
                    try:
                        d.write(t)
                    except Exception:                       # noqa: BLE001
                        pass
            return len(t)

        def flush(self):
            for d in (arquivo, console):
                if d is not None:
                    try:
                        d.flush()
                    except Exception:                       # noqa: BLE001
                        pass

        def isatty(self):
            return False

    sys.stdout = sys.stderr = _Espelho()
    try:
        codigo = main()
    except BaseException:                                   # noqa: BLE001
        import traceback
        traceback.print_exc()
        codigo = 1
    finally:
        arquivo.close()
    sys.exit(codigo)
