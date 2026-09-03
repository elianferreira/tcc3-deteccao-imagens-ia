"""Mede a FPR de uma tecnica fonte real a fonte real.

O que este script descobriu, em 31/08/2026
------------------------------------------
Ele foi escrito para medir a FPR da T01 sobre as "quatro fontes reais da
amostra" -- fodb, imagenet, open_images e raise --, seguindo o proximo passo
registrado em ``documentacao/ESTADO_ATUAL.md``. Na primeira execucao o **controle
falhou**: a COCO, que deveria reproduzir a FPR de 0,0233 reportada no
Capitulo 4, deu 0,6731.

A causa e que ``data/amostra`` **nao contem imagens reais**. E o corpus
sintetico de verificacao funcional produzido por ``make_sample_corpus.py``:
texturas de ruido 1/f em ``real/`` e texturas com artefato periodico plantado
em ``fake/``. As pastas levam os nomes de ``REAL_SOURCES`` porque o gerador as
nomeia assim, e nao ha nada em disco alem delas com esses nomes -- o corpus
deste trabalho tem **uma unica fonte real, a COCO**.

Por isso este script agora recusa qualquer diretorio marcado com
``AVISO_CORPUS_SINTETICO.txt``, em vez de medir e reportar um numero sem
sentido.

Para que ele ainda serve
------------------------
A pergunta continua legitima e continua sem resposta: a FPR de 0,0233 da T01 e
medida contra a COCO, a unica fonte real que ela viu no treinamento, e nao se
sabe qual seria contra outras. Responder exige **baixar imagens reais de
verdade** de FODB, ImageNet, Open Images ou RAISE. No dia em que existirem em
disco, este script as mede sem alteracao::

    python automacao/medir_fpr_t01_fontes.py --amostra data/<corpus real>/real

Condicao de medicao
-------------------
As imagens devem estar na condicao em que o Capitulo 4 foi medido: 256x256
RGB, normalizadas por ``resize_and_center_crop``. O script verifica e avisa
quando nao estao, mas nao normaliza -- normalizar em silencio esconderia
exatamente o tipo de divergencia que originou este arquivo.

Saidas
------
``resultados/fpr_t01_fontes/escores.csv``  -- um escore por imagem
``resultados/fpr_t01_fontes/resumo.csv``   -- FPR por fonte, com IC de Wilson
``resultados/fpr_t01_fontes/resumo.json``  -- o mesmo, com metadados da execucao
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

RAIZ = Path(__file__).resolve().parents[1]
if str(RAIZ) not in sys.path:
    sys.path.insert(0, str(RAIZ))

from codigo.configuracao import DECISION_THRESHOLD, WEIGHTS_DIR  # noqa: E402

FONTES = ("coco", "fodb", "imagenet", "open_images", "raise")
EXTENSOES = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
MARCADOR_SINTETICO = "AVISO_CORPUS_SINTETICO.txt"


def recusar_corpus_sintetico(diretorio: Path) -> None:
    """Aborta se o diretorio -- ou algum ancestral -- for corpus de teste.

    Esta e a trava que faltava em 31/08/2026. O marcador e escrito por
    ``make_sample_corpus.py``; a busca sobe a arvore porque o caminho passado
    costuma ser ``<corpus>/real``, e o marcador fica na raiz do corpus.
    """
    for candidato in [diretorio, *diretorio.parents]:
        marcador = candidato / MARCADOR_SINTETICO
        if marcador.exists():
            raise SystemExit(
                f"RECUSADO: {diretorio} pertence a um corpus sintetico de teste.\n"
                f"  marcador: {marcador}\n"
                "  As pastas de real/ sao ruido procedural, nao fotografias, e\n"
                "  metricas medidas sobre elas nao tem valor cientifico.\n"
                "  Aponte --amostra para um corpus de imagens reais."
            )


def conferir_condicao(caminhos: list[Path], esperado: int = 256) -> str:
    """Confere que as imagens estao na condicao medida no Capitulo 4."""
    from PIL import Image

    divergentes = []
    for caminho in caminhos:
        with Image.open(caminho) as imagem:
            if imagem.size != (esperado, esperado) or imagem.mode != "RGB":
                divergentes.append(f"{caminho.name}:{imagem.size}:{imagem.mode}")
    if divergentes:
        return (f"{len(divergentes)}/{len(caminhos)} fora de {esperado}x{esperado} RGB "
                f"(ex.: {divergentes[0]}) -- NAO normalizadas por este script")
    return "ok"



def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """IC de Wilson para uma proporcao. Com n = 52 o IC normal e inutilizavel
    nos extremos (k = 0 ou k = n dariam largura zero)."""
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    denom = 1.0 + z * z / n
    centro = (p + z * z / (2 * n)) / denom
    margem = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, centro - margem), min(1.0, centro + margem))


def listar(diretorio: Path) -> list[Path]:
    return sorted(p for p in diretorio.iterdir() if p.suffix.lower() in EXTENSOES)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--amostra", type=Path, default=RAIZ / "data" / "amostra" / "real")
    ap.add_argument("--pesos", type=Path, default=WEIGHTS_DIR / "t01_cooccurrence.pt")
    ap.add_argument("--limiar", type=float, default=DECISION_THRESHOLD)
    ap.add_argument("--saida", type=Path, default=RAIZ / "resultados" / "fpr_t01_fontes")
    ap.add_argument("--manifesto", type=Path, default=None,
                    help="desagrega a FPR por split; serve de controle de "
                         "reproducao do valor reportado no Capitulo 4")
    ap.add_argument("--dispositivo", default=None)
    args = ap.parse_args()

    recusar_corpus_sintetico(args.amostra)

    from codigo.tecnicas.t01_coocorrencia import T01Cooccurrence

    t01 = T01Cooccurrence(device=args.dispositivo)
    t01.load(args.pesos)
    print(f"T01 carregada de {args.pesos} em {t01.device}", flush=True)

    args.saida.mkdir(parents=True, exist_ok=True)
    linhas: list[dict] = []
    resumo: list[dict] = []

    for fonte in FONTES:
        diretorio = args.amostra / fonte
        if not diretorio.is_dir():
            print(f"AVISO: {diretorio} nao existe, fonte ignorada", flush=True)
            continue
        recusar_corpus_sintetico(diretorio)
        caminhos = listar(diretorio)
        condicao = conferir_condicao(caminhos)
        if condicao != "ok":
            print(f"AVISO {fonte}: {condicao}", flush=True)
        inicio = datetime.now()
        # num_workers=0: sao 52 imagens, o arranque dos workers custaria mais
        # que a inferencia inteira.
        escores = t01.predict_proba(caminhos, num_workers=0)
        segundos = (datetime.now() - inicio).total_seconds()

        acusadas = escores >= args.limiar
        k, n = int(acusadas.sum()), int(escores.size)
        baixo, alto = wilson(k, n)

        for caminho, escore in zip(caminhos, escores):
            linhas.append({"fonte": fonte, "arquivo": caminho.name,
                           "escore_t01": float(escore),
                           "acusada": int(escore >= args.limiar)})

        resumo.append({
            "fonte": fonte, "n": n, "falsos_positivos": k,
            "fpr": k / n,
            "ic95_baixo": baixo, "ic95_alto": alto,
            "mediana": float(np.median(escores)),
            "media": float(escores.mean()),
            "p25": float(np.percentile(escores, 25)),
            "p75": float(np.percentile(escores, 75)),
            "ms_por_imagem": 1000.0 * segundos / n,
            "condicao": condicao,
        })
        print(f"{fonte:<14} n={n:>3}  FPR={k / n:.4f}  "
              f"IC95=[{baixo:.3f}; {alto:.3f}]  mediana={np.median(escores):.4f}",
              flush=True)

    # Agregado das quatro fontes fora do treino -- e este o numero que falta no
    # texto, ao lado do 0,0233 medido so contra a COCO.
    # Controle de reproducao: o valor do Capitulo 4 e medido no split `test`.
    if args.manifesto is not None:
        import csv as _csv
        from collections import defaultdict
        por_split = defaultdict(list)
        indice = {l["arquivo"]: l["escore_t01"] for l in linhas}
        with args.manifesto.open(newline="", encoding="utf-8") as fh:
            for registro in _csv.DictReader(fh):
                if registro["label"] != "0":
                    continue
                nome = Path(registro["path"].replace("\\", "/")).stem + ".png"
                if nome in indice:
                    por_split[registro["split"]].append(indice[nome])
        print()
        for split in ("train", "val", "test"):
            valores = np.asarray(por_split.get(split, []), dtype=float)
            if valores.size == 0:
                continue
            k, n = int((valores >= args.limiar).sum()), int(valores.size)
            baixo, alto = wilson(k, n)
            resumo.append({
                "fonte": f"coco/split={split}", "n": n, "falsos_positivos": k,
                "fpr": k / n, "ic95_baixo": baixo, "ic95_alto": alto,
                "mediana": float(np.median(valores)), "media": float(valores.mean()),
                "p25": float(np.percentile(valores, 25)),
                "p75": float(np.percentile(valores, 75)),
                "ms_por_imagem": float("nan"), "condicao": "recorte do manifesto",
            })
            print(f"coco/{split:<10} n={n:>6}  FPR={k / n:.4f}  "
                  f"IC95=[{baixo:.3f}; {alto:.3f}]", flush=True)

    fora = [r for r in resumo if r["fonte"] in FONTES and r["fonte"] != "coco"]
    if fora:
        k = sum(r["falsos_positivos"] for r in fora)
        n = sum(r["n"] for r in fora)
        baixo, alto = wilson(k, n)
        resumo.append({
            "fonte": "AGREGADO_4_FONTES_FORA_DO_TREINO", "n": n,
            "falsos_positivos": k, "fpr": k / n,
            "ic95_baixo": baixo, "ic95_alto": alto,
            "mediana": float("nan"), "media": float("nan"),
            "p25": float("nan"), "p75": float("nan"), "ms_por_imagem": float("nan"),
            "condicao": "",
        })
        print(f"\n{'4 fontes fora do treino':<24} n={n}  FPR={k / n:.4f}  "
              f"IC95=[{baixo:.3f}; {alto:.3f}]", flush=True)

    import csv
    with (args.saida / "escores.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(linhas[0].keys()))
        w.writeheader()
        w.writerows(linhas)
    with (args.saida / "resumo.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(resumo[0].keys()))
        w.writeheader()
        w.writerows(resumo)
    with (args.saida / "resumo.json").open("w", encoding="utf-8") as fh:
        json.dump({
            "executado_em": datetime.now().isoformat(timespec="seconds"),
            "pesos": str(args.pesos), "limiar": args.limiar,
            "dispositivo": t01.device, "amostra": str(args.amostra),
            "resumo": resumo,
        }, fh, ensure_ascii=False, indent=2)

    print(f"\nSaidas em {args.saida}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
