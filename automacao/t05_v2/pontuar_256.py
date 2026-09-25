"""Passo 2 - T02 e T03 em 256 px sobre a particao `test` do protocolo OOD.

Por que esta rodada e necessaria
--------------------------------
Os caches `resultados/scores/ood__T0*__clean.npy` tem **7.138** posicoes e vem
do `manifesto30k_ood_familias.csv`; a particao `test` de `manifesto30k_ood.csv`
tem **14.788**. E a armadilha 7 de ESTADO_ATUAL.md: parear os dois
posicionalmente compararia imagens diferentes.

A particao `fusion` (2.350) **ja esta em cache e completa** para T01/T02/T03/T04
-- e o ajuste nao precisa desta rodada. Isto aqui e so a avaliacao.

Custo declarado: T02 a ~555 ms/imagem => ~2h15 para 14.788. T03 a ~41 ms => ~10
min. Grava incrementalmente, entao uma queda no meio nao perde o que ja saiu.

Escopo: **256 px**, que e a condicao em que a fusao e calibrada. Nao e a
condicao de exibicao da v2 para T02 e T03 -- ver o cabecalho de `preparar.py`
para a medicao que descartou o protocolo nativo.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

RAIZ = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(RAIZ))

SAIDA = RAIZ / "resultados" / "t05_v2"
LOTE_PADRAO = 250


def carregar_alvo() -> pd.DataFrame:
    indice = pd.read_csv(SAIDA / "indice_limpeza.csv")
    return indice[indice["split"] == "test"].reset_index(drop=True)


def pontuar(nome: str, tecnica, caminhos: list[Path], destino: Path,
            lote: int) -> None:
    """Pontua em lotes, gravando a cada lote. Retomavel."""
    feitos: dict[str, float] = {}
    if destino.exists():
        anterior = pd.read_csv(destino)
        feitos = dict(zip(anterior["path"], anterior["escore"]))
        print(f"   {nome}: {len(feitos)} ja pontuadas, retomando")

    pendentes = [c for c in caminhos if str(c) not in feitos]
    if not pendentes:
        print(f"   {nome}: nada a fazer")
        return

    inicio = time.perf_counter()
    for i in range(0, len(pendentes), lote):
        bloco = pendentes[i:i + lote]
        escores = tecnica.predict_proba(bloco)
        for caminho, escore in zip(bloco, np.asarray(escores, dtype=float)):
            feitos[str(caminho)] = float(escore)
        pd.DataFrame(
            {"path": list(feitos), "escore": list(feitos.values())}
        ).to_csv(destino, index=False)
        decorrido = time.perf_counter() - inicio
        prontos = i + len(bloco)
        taxa = decorrido / prontos
        resta = (len(pendentes) - prontos) * taxa
        print(f"   {nome}: {prontos}/{len(pendentes)}  "
              f"{taxa*1000:.0f} ms/img  restam ~{resta/60:.0f} min", flush=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lote", type=int, default=LOTE_PADRAO)
    parser.add_argument("--tecnicas", default="T03,T02",
                        help="ordem de execucao; T03 primeiro por ser barata")
    args = parser.parse_args()

    alvo = carregar_alvo()
    caminhos = [Path(p) for p in alvo["path"]]
    print(f"particao test: {len(caminhos)} imagens em 256 px")

    from codigo.configuracao import WEIGHTS_DIR

    for nome in [t.strip() for t in args.tecnicas.split(",") if t.strip()]:
        destino = SAIDA / f"{nome.lower()}_256_test.csv"
        print(f"\n=== {nome} -> {destino.name}")
        if nome == "T03":
            from codigo.tecnicas.t03_benford import T03Benford
            tecnica = T03Benford().load(WEIGHTS_DIR / "t03_benford.pkl")
        elif nome == "T02":
            # Nao ha `load`: o wrapper resolve repositorio e checkpoint pela
            # ExternalConfig, e roda o SPAI em subprocesso.
            from codigo.tecnicas.t02_spai import T02SPAI
            tecnica = T02SPAI()
        else:
            raise SystemExit(f"tecnica desconhecida: {nome}")
        pontuar(nome, tecnica, caminhos, destino, args.lote)

    print("\nconcluido")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
