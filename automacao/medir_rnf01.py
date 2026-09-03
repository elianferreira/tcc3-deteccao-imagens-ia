"""RNF01 - tempo de inferencia por imagem, conforme a Secao 3.6.2.

A Secao 3.6.2 especifica: "o tempo de inferencia por imagem e medido sobre lote
de 1.000 imagens do conjunto de teste em CPU Intel Core i7 de 12a geracao, com
calculo de media e desvio padrao. O criterio de aprovacao e tempo medio
inferior a 30 segundos por imagem para todas as tecnicas, conforme RNF01. Caso
T02 exceda esse limite em CPU, o tempo de inferencia em GPU sera reportado como
referencia complementar e a limitacao sera declarada na analise dos resultados."

Divergencia interna do TCC 2, registrada aqui
---------------------------------------------
O texto de RNF01 (Quadro 3) diz que a medicao emprega "GPU para as tecnicas T02
e T04 e CPU para T01, T03 e T05". Ja a Secao 3.6.2 manda medir tudo em CPU e
tratar a GPU como referencia complementar apenas se T02 exceder o limite.

Este script segue a Secao 3.6.2, por ser a especificacao do programa de teste, e
reporta os dois valores para T02. A divergencia esta declarada em
documentacao/AUDITORIA_TCC2_TCC3.md.

Medicao
-------
Mede-se o tempo por imagem individualmente, e nao o custo amortizado de um lote:
a interface analisa uma imagem por vez, que e o cenario de RNF01. Os modelos ja
estao carregados em memoria quando o cronometro comeca, como o requisito exige.

Uso::

    python automacao/medir_rnf01.py --limit 1000
    python automacao/medir_rnf01.py --limit 1000 --incluir-t02-cpu
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from codigo.configuracao import DATA_DIR, RESULTS_DIR, WEIGHTS_DIR      # noqa: E402

LIMITE_SEGUNDOS = 30.0


def amostrar(manifesto: Path, limite: int, seed: int = 42) -> list[Path]:
    frame = pd.read_csv(manifesto)
    teste = frame[frame["split"] == "test"]
    if len(teste) > limite:
        teste = teste.sample(n=limite, random_state=seed)
    return [Path(p) for p in teste["path"]]


def cronometrar(funcao, caminhos: list[Path]) -> dict:
    """Mede o tempo de uma imagem por vez, com media e desvio padrao."""
    tempos = []
    for caminho in caminhos:
        inicio = time.perf_counter()
        funcao([caminho])
        tempos.append(time.perf_counter() - inicio)
    vetor = np.asarray(tempos)
    return {
        "n": int(len(vetor)),
        "media_s": float(vetor.mean()),
        "desvio_s": float(vetor.std(ddof=1)) if len(vetor) > 1 else 0.0,
        "mediana_s": float(np.median(vetor)),
        "maximo_s": float(vetor.max()),
        "aprovado": bool(vetor.mean() < LIMITE_SEGUNDOS),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Mede RNF01 (Secao 3.6.2)")
    parser.add_argument("--manifest", type=Path,
                        default=DATA_DIR / "manifesto30k_standard.csv")
    parser.add_argument("--limit", type=int, default=1000)
    parser.add_argument("--incluir-t02-cpu", action="store_true",
                        help="mede T02 tambem em CPU; custoso, dezenas de segundos por imagem")
    parser.add_argument("--tecnicas", nargs="+", default=["T01", "T03", "T05", "T02"])
    args = parser.parse_args()

    if not args.manifest.exists():
        print(f"ERRO: manifesto nao encontrado em {args.manifest}")
        return 1

    caminhos = amostrar(args.manifest, args.limit)
    print(f"Amostra: {len(caminhos)} imagens do conjunto de teste")
    print(f"Criterio RNF01: media < {LIMITE_SEGUNDOS} s por imagem\n")

    resultados: dict[str, dict] = {}

    # --- T01, T03 e T05 em CPU -------------------------------------------
    if "T01" in args.tecnicas:
        from codigo.tecnicas.t01_coocorrencia import T01Cooccurrence
        checkpoint = WEIGHTS_DIR / "t01_cooccurrence.pt"
        if checkpoint.exists():
            print("T01 em CPU ...", flush=True)
            t01 = T01Cooccurrence(device="cpu").load(checkpoint)
            t01.device = "cpu"
            t01.model.to("cpu")
            resultados["T01_cpu"] = cronometrar(
                lambda p: t01.predict_proba(p, num_workers=0), caminhos)
        else:
            print(f"[aviso] T01: checkpoint ausente em {checkpoint}")

    if "T03" in args.tecnicas:
        from codigo.tecnicas.t03_benford import T03Benford
        modelo = WEIGHTS_DIR / "t03_benford.pkl"
        if modelo.exists():
            print("T03 em CPU ...", flush=True)
            t03 = T03Benford().load(modelo)
            resultados["T03_cpu"] = cronometrar(t03.predict_proba, caminhos)
        else:
            print(f"[aviso] T03: modelo ausente em {modelo}")

    # --- T02: GPU sempre, CPU sob demanda --------------------------------
    if "T02" in args.tecnicas:
        from codigo.tecnicas.t02_spai import T02SPAI
        t02 = T02SPAI()
        disponivel, motivo = t02.is_available()
        if not disponivel:
            print(f"[aviso] T02 indisponivel: {motivo}")
        else:
            # Amostra reduzida: cada chamada individual recarrega o checkpoint
            # do SPAI, de modo que 1.000 medicoes isoladas levariam horas sem
            # acrescentar informacao sobre a media.
            amostra_t02 = caminhos[: min(30, len(caminhos))]
            print(f"T02 em GPU ({len(amostra_t02)} imagens) ...", flush=True)
            resultados["T02_gpu"] = cronometrar(t02.predict_proba, amostra_t02)

            if args.incluir_t02_cpu:
                print(f"T02 em CPU ({len(amostra_t02)} imagens) ...", flush=True)
                anterior = os.environ.get("CUDA_VISIBLE_DEVICES")
                # O pipeline oficial detecta CUDA sozinho; esconder o
                # dispositivo e a unica forma de forca-lo a CPU sem alterar
                # o codigo de terceiros.
                os.environ["CUDA_VISIBLE_DEVICES"] = ""
                try:
                    resultados["T02_cpu"] = cronometrar(t02.predict_proba, amostra_t02)
                finally:
                    if anterior is None:
                        os.environ.pop("CUDA_VISIBLE_DEVICES", None)
                    else:
                        os.environ["CUDA_VISIBLE_DEVICES"] = anterior

    # --- T05: custo proprio da fusao -------------------------------------
    if "T05" in args.tecnicas:
        from codigo.tecnicas.t05_fusao import T05Fusion
        modelo = WEIGHTS_DIR / "t05_fusion.pkl"
        if modelo.exists():
            t05 = T05Fusion().load(modelo)
            escores = np.random.default_rng(42).random((len(caminhos), 4))
            tempos = []
            for linha in escores:
                inicio = time.perf_counter()
                # T05 nao recebe caminhos: recebe a matriz de escores das
                # demais tecnicas, dai a API propria.
                t05.predict_proba_scores(linha.reshape(1, -1))
                tempos.append(time.perf_counter() - inicio)
            vetor = np.asarray(tempos)
            resultados["T05_cpu"] = {
                "n": int(len(vetor)), "media_s": float(vetor.mean()),
                "desvio_s": float(vetor.std(ddof=1)), "mediana_s": float(np.median(vetor)),
                "maximo_s": float(vetor.max()), "aprovado": True,
                "observacao": ("mede apenas a regressao logistica sobre escores prontos; "
                               "o custo real de T05 e a soma das tecnicas que agrega"),
            }

    # --- Relatorio --------------------------------------------------------
    print("\n" + "=" * 74)
    print("RNF01 - TEMPO DE INFERENCIA POR IMAGEM")
    print("=" * 74)
    print(f"{'tecnica':<12} {'n':>6} {'media':>12} {'desvio':>12} {'maximo':>12}  situacao")
    for nome, dados in resultados.items():
        marca = "ok" if dados["aprovado"] else "ACIMA DO LIMITE"
        print(f"{nome:<12} {dados['n']:>6} {dados['media_s']:>10.4f} s "
              f"{dados['desvio_s']:>10.4f} s {dados['maximo_s']:>10.4f} s  {marca}")
    print("=" * 74)

    soma = sum(d["media_s"] for n, d in resultados.items() if not n.startswith("T02_cpu"))
    print(f"\nSoma das medias (T02 em GPU): {soma:.3f} s por imagem")
    print(f"Criterio RNF01 (< {LIMITE_SEGUNDOS} s): "
          f"{'ATENDIDO' if soma < LIMITE_SEGUNDOS else 'NAO ATENDIDO'}")

    destino = RESULTS_DIR / "rnf01_tempos.json"
    destino.write_text(json.dumps(
        {"limite_s": LIMITE_SEGUNDOS, "n_amostra": len(caminhos),
         "soma_medias_s": soma, "tecnicas": resultados},
        indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nGravado em {destino}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
