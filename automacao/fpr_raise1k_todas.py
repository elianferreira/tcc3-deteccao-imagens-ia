"""FPR das cinco tecnicas sobre o RAISE-1k — a medicao que fecha a A3.

O problema
----------
Em 07/09/2026 mediu-se que a T01 da replica desaba quando a classe real sai da
distribuicao: FPR de 3,9% (COCO) para 60,4% (RAISE-1k), e AUC de 0,824 para
0,392 — abaixo do acaso. Ver `O protocolo OOD era meio OOD` no cofre.

Aquilo foi medido no modelo da **replica**. As tecnicas do trabalho principal —
e sobretudo a **T05**, que e a contribuicao proposta — nao foram medidas na
fonte real nova. Enquanto nao forem, nao se pode dizer nem que a FPR da T05 caiu
nem que ela resistiu.

O controle e obrigatorio
------------------------
Este script mede **duas** amostras de mesmo tamanho:

  * `raise1k` — as 999 imagens normalizadas em `data/raise1k_norm/`
  * `coco`    — 999 reais do COCO, sorteadas do split de teste

O COCO e o **controle**: se a FPR dele nao reproduzir a ordem de grandeza
publicada, a medicao esta errada e o numero do RAISE nao significa nada. Foi
exatamente um controle assim que derrubou o falso achado de vies da T01 em
31/08 (`medir_fpr_t01_fontes.py`), quando a COCO deu 0,6731 em vez de 0,0233.

T04 — e a retratacao do motivo pelo qual ela ficava de fora
----------------------------------------------------------
⛔ **Ate 23/09/2026 este arquivo dizia que os extratores da T04 estavam sob
"bloqueio externo" e mandava a coluna dela como NaN. O motivo era falso.** Ele
foi copiado de `externo/CONTRATO.md`, mas a sessao de **07/09** — o mesmo dia
desta medicao — ja registrava que o bloqueio tinha sido resolvido em 30/08: os
servicos residentes do WSL2 pontuam imagem arbitraria sob demanda.

A consequencia nao e cosmetica. O numero que saiu daqui ("a T05 vai de 4,3% para
47,4% de falso positivo") e da fusao de **tres** fontes, nao da de quatro que o
sistema entrega — e e ele que sustenta a A3. Alem disso, a T04 e, junto com a
T02, uma das **duas** tecnicas de pesos oficiais do trabalho; excluindo-a, a
tese de que "so a tecnica de pesos congelados atravessa" ficou apoiada numa
unica instancia.

Com `--com-t04` a coluna entra medida, lida dos CSVs que
`automacao/extrair_t04_raise1k.py` produz. Sem a opcao, o comportamento antigo
e preservado — a rodada de 07/09 continua reproduzivel.

Uso::

    python automacao/fpr_raise1k_todas.py --com-t04
    python automacao/fpr_raise1k_todas.py --n 200      # ensaio rapido
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

RAIZ = Path(__file__).resolve().parents[1]
if str(RAIZ) not in sys.path:
    sys.path.insert(0, str(RAIZ))

from codigo.configuracao import DECISION_THRESHOLD, WEIGHTS_DIR  # noqa: E402

RAISE_DIR = RAIZ / "data" / "raise1k_norm" / "real" / "raise1k"
MANIFESTO_OOD = RAIZ.parent / "tcc3-t01-replicacao" / "data" / "manifesto_ood13.csv"
SAIDA = RAIZ / "resultados" / "fpr_raise1k"

# ⚠️ `--sufixo` existe por causa da armadilha que ja custou os numeros da
# dissertacao uma vez: `avaliar_t04_componentes.py` gravava sempre no mesmo
# caminho e sobrescreveu os resultados publicados na primeira execucao de uma
# variante. Aqui cada variante de peso escreve na propria pasta.
SAIDA_BASE = SAIDA
COFRE = Path(r"C:\Users\ferre\OneDrive\Documentos\TCC 3")

# Referencias publicadas, protocolo padrao (Numeros para citar).
PUBLICADO = {"T01": 0.0233, "T02": 0.0504, "T05": 0.0042}


def _wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    den = 1 + z * z / n
    centro = (p + z * z / (2 * n)) / den
    meio = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return (max(0.0, centro - meio), min(1.0, centro + meio))


def imagens_raise(n: int | None) -> list[Path]:
    todas = sorted(p for p in RAISE_DIR.glob("*.png") if "__recorte" not in p.name)
    if not todas:
        raise SystemExit(f"ERRO: nenhuma imagem em {RAISE_DIR}")
    return todas[:n] if n else todas


def imagens_coco(quantidade: int, semente: int = 42) -> list[Path]:
    reais = []
    with open(MANIFESTO_OOD, newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            if int(r["label"]) == 0:
                reais.append(RAIZ / r["path"].replace("\\", "/"))
    existentes = [p for p in reais if p.exists()]
    if len(existentes) < quantidade:
        raise SystemExit(f"ERRO: so {len(existentes)} reais do COCO em disco")
    random.Random(semente).shuffle(existentes)
    return existentes[:quantidade]


def escorear_tecnicas(caminhos: list[Path], rotulo: str) -> dict[str, np.ndarray]:
    """Escores de T01, T02 e T03.

    A T04 nao entra aqui porque nao roda no mesmo processo: ela vem dos
    servicos do WSL2, numa passagem anterior, e e lida por `escorear_t04`.
    """
    escores: dict[str, np.ndarray] = {}

    from codigo.tecnicas.t01_coocorrencia import T01Cooccurrence
    inicio = time.time()
    t01 = T01Cooccurrence(device="cuda")
    t01.load(WEIGHTS_DIR / "t01_cooccurrence.pt")
    p = np.asarray(t01.predict_proba(caminhos), dtype=np.float64)
    escores["T01"] = p[:, 1] if p.ndim > 1 else p
    print(f"  [{rotulo}] T01 em {(time.time()-inicio)/60:.1f} min", flush=True)

    from codigo.tecnicas.t03_benford import T03Benford
    inicio = time.time()
    t03 = T03Benford(n_jobs=4).load(WEIGHTS_DIR / "t03_benford.pkl")
    p = np.asarray(t03.predict_proba(caminhos), dtype=np.float64)
    escores["T03"] = p[:, 1] if p.ndim > 1 else p
    print(f"  [{rotulo}] T03 em {(time.time()-inicio)/60:.1f} min", flush=True)

    from codigo.tecnicas.t02_spai.tecnica import T02SPAI
    inicio = time.time()
    t02 = T02SPAI()
    disponivel, motivo = t02.is_available()
    if not disponivel:
        print(f"  [{rotulo}] T02 INDISPONIVEL: {motivo}", flush=True)
    else:
        try:
            p = np.asarray(t02.predict_proba(caminhos), dtype=np.float64)
            escores["T02"] = p[:, 1] if p.ndim > 1 else p
            print(f"  [{rotulo}] T02 em {(time.time()-inicio)/60:.1f} min", flush=True)
        except Exception as erro:                          # noqa: BLE001
            print(f"  [{rotulo}] T02 FALHOU: {erro}", flush=True)

    return escores


def escorear_t05(escores: dict[str, np.ndarray], n: int) -> np.ndarray | None:
    from codigo.tecnicas.t05_fusao.tecnica import T05Fusion, build_score_matrix
    caminho = WEIGHTS_DIR / "t05_fusion.pkl"
    if not caminho.exists():
        print(f"  T05: pesos ausentes em {caminho}", flush=True)
        return None
    try:
        t05 = T05Fusion().load(caminho)
        matriz = build_score_matrix(escores, n)
        p = np.asarray(t05.predict_proba_scores(matriz), dtype=np.float64)
        return p[:, 1] if p.ndim > 1 else p
    except Exception as erro:                              # noqa: BLE001
        print(f"  T05 FALHOU: {erro}", flush=True)
        return None


RAISE_T04 = RAIZ / "resultados" / "t04_escores_raise1k.csv"
COCO_T04 = RAIZ / "resultados" / "t04_escores_coco_controle.csv"
COMPONENTES_T04 = ("object_shadow", "perspective_fields", "line_segment")


def _ler_componentes(caminho: Path) -> dict[str, list[float]]:
    """arquivo -> [objeto_sombra, campos, retas], com ausencia virando NaN."""
    if not caminho.exists():
        return {}
    tabela: dict[str, list[float]] = {}
    with caminho.open(encoding="utf-8", newline="") as arquivo:
        for linha in csv.DictReader(arquivo):
            valores = []
            for componente in COMPONENTES_T04:
                bruto = (linha.get(componente) or "").strip()
                valores.append(float(bruto) if bruto else float("nan"))
            tabela[linha["arquivo"]] = valores
    return tabela


def escorear_t04(caminhos: list[Path], rotulo: str) -> np.ndarray | None:
    """T04 lida de CSV, nao dos servicos — e por que as duas coisas importam.

    **Por que de CSV:** os servicos do WSL2 ocupam a placa e nao convivem com a
    T02 nos 6 GB da RTX 3060 (armadilha nº 8). A extracao roda numa passagem
    anterior, os servicos caem, e esta passagem le o resultado. Ver
    `automacao/extrair_t04_raise1k.py`.

    ⛔ **Por que nao do cache do corpus:** `t04_escores_componentes_ood_completo.csv`
    cobre `coco_000000` a `coco_029999`, e as 999 reais do controle vem de
    `corvi2024_escala`, onde os nomes passam de `coco_058406`. Conferido em
    23/09: o cache **nao** cobre o controle, e parear por nome teria casado
    imagens diferentes em silencio (armadilha nº 5). Os dois conjuntos sao
    extraidos pelos mesmos servicos, na mesma rodada.

    O escore da linha e o ``nanmean`` das tres representacoes, igual ao
    adaptador de `codigo/tecnicas/t04_geometria/tecnica.py`.
    """
    origem = RAISE_T04 if rotulo == "raise1k" else COCO_T04
    tabela = _ler_componentes(origem)
    if not tabela:
        print(f"  [{rotulo}] T04 INDISPONIVEL: {origem.name} ausente ou vazio",
              flush=True)
        return None

    faltantes = [c.stem for c in caminhos if c.stem not in tabela]
    if faltantes:
        print(f"  [{rotulo}] T04: {len(faltantes)} imagens sem escore "
              f"(entram como NaN; ex.: {faltantes[0]})", flush=True)

    linhas = [tabela.get(c.stem, [float("nan")] * 3) for c in caminhos]
    with np.errstate(invalid="ignore"):
        medias = np.nanmean(np.asarray(linhas, dtype=np.float64), axis=1)
    validos = int(np.isfinite(medias).sum())
    print(f"  [{rotulo}] T04 de {origem.name}: {validos}/{len(caminhos)} "
          f"com escore", flush=True)
    return medias


def medir(caminhos: list[Path], rotulo: str, com_t04: bool = False) -> dict:
    print(f"\n=== {rotulo}: {len(caminhos)} imagens ===", flush=True)
    escores = escorear_tecnicas(caminhos, rotulo)
    if com_t04:
        t04 = escorear_t04(caminhos, rotulo)
        if t04 is not None and np.isfinite(t04).any():
            escores["T04"] = t04
    t05 = escorear_t05(escores, len(caminhos))
    if t05 is not None:
        escores["T05"] = t05

    resultado = {}
    for tecnica, valores in escores.items():
        # ⚠️ A T04 pode trazer NaN (imagem sem escore de alguma representacao).
        # Contar sobre `len(valores)` diluiria a FPR pelo que nao foi medido --
        # o denominador e o que tem escore, e o n reportado diz quanto e.
        finitos = valores[np.isfinite(valores)]
        if finitos.size == 0:
            print(f"  [{rotulo}] {tecnica}: nenhum escore finito, fora do resumo",
                  flush=True)
            continue
        fp = int((finitos > DECISION_THRESHOLD).sum())
        baixo, alto = _wilson(fp, finitos.size)
        resultado[tecnica] = {
            "n": int(finitos.size), "n_enviadas": len(valores),
            "falsos_positivos": fp,
            "fpr": fp / finitos.size, "ic_baixo": baixo, "ic_alto": alto,
            "escore_mediano": float(np.median(finitos)),
        }
        np.save(SAIDA / f"escores__{rotulo}__{tecnica}.npy", valores)
    return resultado


def relatar(res: dict) -> str:
    agora = datetime.now()
    com_t04 = "T04" in res.get("raise1k", {})
    if com_t04:
        aviso = (
            "> **A T04 entra medida nas duas amostras** — RAISE de "
            "`t04_escores_raise1k.csv` (extraido nesta rodada), COCO do cache do "
            "corpus. **Esta e a T05 de quatro fontes**, a mesma do sistema "
            "entregue, e por isso comparavel com o numero publicado.")
    else:
        aviso = (
            "> **A T04 entra como NaN nas duas amostras**, entao a T05 aqui e a "
            "de tres fontes — comparavel entre as duas colunas, nao com o "
            "numero publicado. Rode com `--com-t04` para a de quatro.")
    linhas = [
        "---", "tipo: resultados", "tags: [tcc3, resultados, corpus]",
        f"atualizado: {agora:%Y-%m-%d}", "---", "",
        "# FPR das tecnicas no RAISE-1k", "",
        f"> Gerado por `automacao/fpr_raise1k_todas.py` em "
        f"{agora:%d/%m/%Y as %H:%M}. Limiar {DECISION_THRESHOLD}.",
        aviso, "",
        "## FPR, fonte real do treino (COCO) x fonte nova (RAISE-1k)", "",
        "| tecnica | COCO (controle) | RAISE-1k | razao | publicado (padrao) |",
        "|---|---|---|---|---|",
    ]
    coco, raise1k = res.get("coco", {}), res.get("raise1k", {})
    for tecnica in ("T01", "T02", "T03", "T04", "T05"):
        c, r = coco.get(tecnica), raise1k.get(tecnica)
        if not c and not r:
            continue
        cs = (f"{c['fpr']:.4f} [{c['ic_baixo']:.3f};{c['ic_alto']:.3f}]"
              if c else "—")
        rs = (f"**{r['fpr']:.4f}** [{r['ic_baixo']:.3f};{r['ic_alto']:.3f}]"
              if r else "—")
        razao = (f"**{r['fpr']/c['fpr']:.1f}x**"
                 if c and r and c["fpr"] > 0 else "—")
        pub = f"{PUBLICADO[tecnica]:.4f}" if tecnica in PUBLICADO else "—"
        linhas.append(f"| {tecnica} | {cs} | {rs} | {razao} | {pub} |")

    linhas += ["", "## Guarda de controle", ""]
    t01c = coco.get("T01")
    if t01c:
        ok = abs(t01c["fpr"] - PUBLICADO["T01"]) < 0.05
        linhas.append(
            f"{'🟢' if ok else '🔴'} FPR da T01 no COCO: **{t01c['fpr']:.4f}** "
            f"contra {PUBLICADO['T01']:.4f} publicado. "
            + ("Dentro da ordem de grandeza — a medicao esta calibrada."
               if ok else "**FORA** do esperado: o numero do RAISE nao vale ate "
                          "entender a divergencia.")
        )
    linhas += ["", "⚠️ **Leitura de sentido nao entra aqui** — o script so mede. "
               "Ver a nota de sessao do dia.", ""]
    return "\n".join(linhas)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=None, help="limitar a N imagens")
    parser.add_argument("--sem-cofre", action="store_true")
    parser.add_argument(
        "--sufixo", default="",
        help="grava em resultados/fpr_raise1k<sufixo>/ em vez de sobrescrever "
             "a rodada anterior. Obrigatorio ao trocar TCC3_WEIGHTS_DIR.")
    parser.add_argument(
        "--com-t04", action="store_true",
        help="inclui a T04 a partir dos CSVs, fazendo da T05 a fusao de quatro "
             "fontes. Exige `extrair_t04_raise1k.py` ja rodado.")
    args = parser.parse_args(argv)

    global SAIDA
    if args.sufixo:
        SAIDA = SAIDA_BASE.with_name(SAIDA_BASE.name + args.sufixo)
    SAIDA.mkdir(parents=True, exist_ok=True)
    print(f"gravando em {SAIDA.name}/ · pesos de {WEIGHTS_DIR.name}/", flush=True)
    print(f"=== inicio {datetime.now():%Y-%m-%d %H:%M:%S} ===", flush=True)

    caminhos_raise = imagens_raise(args.n)
    caminhos_coco = imagens_coco(len(caminhos_raise))

    res = {
        "coco": medir(caminhos_coco, "coco", args.com_t04),
        "raise1k": medir(caminhos_raise, "raise1k", args.com_t04),
        "limiar": DECISION_THRESHOLD,
        "com_t04": args.com_t04,
        "gerado_em": datetime.now().isoformat(),
    }
    (SAIDA / "resumo.json").write_text(
        json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")

    texto = relatar(res)
    (SAIDA / "relatorio.md").write_text(texto, encoding="utf-8")
    if not args.sem_cofre and COFRE.exists():
        (COFRE / "Resultados" / "FPR das tecnicas no RAISE-1k.md").write_text(
            texto, encoding="utf-8")
    print("\n" + texto, flush=True)
    print(f"=== fim {datetime.now():%Y-%m-%d %H:%M:%S} ===", flush=True)
    return 0


def _abrir_log():
    """Abre o log SEMPRE — a guarda condicional ja falhou sob o Agendador."""
    caminho = RAIZ / "logs" / "fpr_raise1k_todas.log"
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
