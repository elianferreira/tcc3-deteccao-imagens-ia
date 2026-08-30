"""Reune os escores das tres representacoes de T04 em um unico arquivo.

Cada componente e extraido por um caminho proprio, em momentos diferentes:

===================  ==========================================  ==============
Representacao        Extrator                                    Onde roda
===================  ==========================================  ==============
object_shadow        SSISv2 -> mapas -> classificador oficial    WSL2 + Windows
perspective_fields   PerspectiveFields -> classificador oficial  WSL2
line_segment         DeepLSD -> PointNet oficial                 WSL2
===================  ==========================================  ==============

Este script junta as tres saidas em ``results/t04_escores_componentes.csv``, com
uma coluna por representacao, que e o formato que
``src/techniques/t04_geometry.py`` consome pela variavel ``TCC3_T04_ESCORES``.

Por que um arquivo so
---------------------
O adaptador precisa dos tres escores alinhados por imagem para agrega-los por
``nanmean``, que e o mecanismo de RN07. Manter tres arquivos separados exigiria
que ele conhecesse o layout de cada extrator; um arquivo com uma coluna por
componente mantem essa complexidade aqui, onde ela e visivel.

Ausencia e preservada como ausencia. Uma imagem sem escore de um componente
recebe vazio na coluna, que vira NaN na leitura, e ``nanmean`` a ignora -- em vez
de imputar um valor que fingiria evidencia inexistente.

Uso::

    python scripts/consolidar_escores_t04.py --variante combined
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import DATA_DIR, RESULTS_DIR      # noqa: E402

# Componente -> (padrao de busca, coluna de escore na origem)
FONTES = {
    "object_shadow": (RESULTS_DIR, "t04_escores_{variante}{sufixo}.csv", "escore_t04"),
    "perspective_fields": (DATA_DIR / "t04_perspective_fields",
                           "{conjunto}_pf_{variante}*.csv", "escore_t04_pf"),
    "line_segment": (DATA_DIR / "t04_line_segments",
                     "{conjunto}_ls_{variante}*.csv", "escore_t04_ls"),
}


def carregar(componente: str, raiz: Path, padrao: str, coluna: str,
             variante: str, conjunto: str, sufixo: str) -> pd.DataFrame | None:
    """Le e concatena os CSV de um componente, sobre todos os splits.

    O filtro por ``conjunto`` nao e cosmetico. O glob antigo varria o
    diretorio inteiro, de modo que uma segunda rodada sobre outro corpus --
    o benchmark de geradores modernos, por exemplo -- seria concatenada a
    rodada padrao **em silencio**, misturando dois corpora num arquivo so.
    """
    alvo = padrao.format(variante=variante, conjunto=conjunto, sufixo=sufixo)
    caminhos = sorted(raiz.glob(alvo)) if "*" in alvo else [raiz / alvo]
    caminhos = [c for c in caminhos if c.exists() and "falhas" not in c.name]
    if not caminhos:
        print(f"  {componente:<20} ausente ({raiz / alvo})")
        return None

    partes = []
    for caminho in caminhos:
        tabela = pd.read_csv(caminho)
        if coluna not in tabela.columns:
            print(f"  [aviso] {caminho.name} sem coluna {coluna}; ignorado")
            continue
        partes.append(tabela[["arquivo", "split", "label", coluna]])

    if not partes:
        return None

    junto = pd.concat(partes, ignore_index=True)
    # Uma imagem pode aparecer em mais de um arquivo se um split foi
    # reprocessado; a ultima extracao prevalece.
    junto = junto.drop_duplicates(subset="arquivo", keep="last")
    junto = junto.rename(columns={coluna: componente})
    print(f"  {componente:<20} {len(junto)} imagens  ({len(caminhos)} arquivo(s))")
    return junto


def main() -> int:
    parser = argparse.ArgumentParser(description="Consolida os escores de T04")
    parser.add_argument("--variante", default="combined",
                        choices=["combined", "indoor", "outdoor"])
    parser.add_argument("--saida", type=Path,
                        default=RESULTS_DIR / "t04_escores_componentes.csv")
    parser.add_argument("--conjunto", default="tcc3_30k",
                        help="etiqueta da rodada; separa corpora distintos")
    parser.add_argument("--sufixo", default="",
                        help="sufixo do CSV de objeto-sombra (ver "
                             "avaliar_t04_corpus.py --sufixo)")
    args = parser.parse_args()

    print("=" * 70)
    print("CONSOLIDACAO DOS ESCORES DE T04")
    print("=" * 70)
    print(f"Variante de pesos: {args.variante}\n")

    tabelas = {}
    for componente, (raiz, padrao, coluna) in FONTES.items():
        tabela = carregar(componente, raiz, padrao, coluna, args.variante,
                          args.conjunto, args.sufixo)
        if tabela is not None:
            tabelas[componente] = tabela

    if not tabelas:
        print("\nERRO: nenhum componente encontrado")
        return 1

    consolidado = None
    for componente, tabela in tabelas.items():
        if consolidado is None:
            consolidado = tabela
            continue
        consolidado = consolidado.merge(
            tabela.drop(columns=["split", "label"]), on="arquivo", how="outer")

    # Splits e rotulos podem ficar vazios em linhas vindas so de um componente.
    for coluna in ("split", "label"):
        if consolidado[coluna].isna().any():
            for tabela in tabelas.values():
                mapa = tabela.set_index("arquivo")[coluna]
                consolidado[coluna] = consolidado[coluna].fillna(
                    consolidado["arquivo"].map(mapa))

    presentes = [c for c in FONTES if c in consolidado.columns]
    consolidado = consolidado[["arquivo", "split", "label"] + presentes]
    consolidado = consolidado.sort_values(["split", "arquivo"]).reset_index(drop=True)

    print("\n" + "=" * 70)
    print(f"Total de imagens: {len(consolidado)}")
    for componente in presentes:
        n = int(consolidado[componente].notna().sum())
        print(f"  {componente:<20} {n:>6} escores  "
              f"({n / len(consolidado):.1%} de cobertura)")

    completos = int(consolidado[presentes].notna().all(axis=1).sum())
    print(f"\nCom as {len(presentes)} representacoes: {completos} imagens "
          f"({completos / len(consolidado):.1%})")

    args.saida.parent.mkdir(parents=True, exist_ok=True)
    consolidado.to_csv(args.saida, index=False)
    print(f"\nGravado em {args.saida}")
    print("Aponte TCC3_T04_ESCORES para este arquivo.")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
