"""Avalia T04 sobre o corpus deste trabalho, a partir dos mapas objeto-sombra.

Fecha a cadeia de T04. O extrator do WSL2
(``automacao/wsl/extrair_object_shadow_ssis.py``) converte cada imagem do corpus
no par de mapas que o classificador oficial consome; este script alimenta o
classificador com esses pares e produz os escores.

Diferenca em relacao a ``replicar_t04_object_shadow.py``
--------------------------------------------------------
Aquele script avalia o classificador sobre as mascaras que os **autores**
publicaram, do corpus **deles** (Kandinsky), e serve para verificar que a
replicacao reproduz o valor publicado. Ele nao produz um numero comparavel as
demais tecnicas.

Este avalia sobre o corpus **deste trabalho** (Corvi et al., 2024), com mapas
extraidos aqui. E o que permite T04 entrar na tabela comparativa e servir de
quarta fonte para a fusao T05, que hoje recebe NaN naquela coluna
(``codigo/tecnicas/t05_fusao/tecnica.py:34``).

Transferencia entre dominios
----------------------------
Os pesos oficiais foram treinados em Kandinsky e sao aplicados aqui sem
reajuste, em conformidade com a Etapa 2 do TCC 2, que veda retreinar as
tecnicas de terceiros. O resultado e, portanto, de transferencia direta: um
desempenho inferior ao publicado pelos autores nao indica falha de replicacao,
e sim mudanca de dominio. Os tres conjuntos de pesos sao avaliados --
``combined``, ``indoor`` e ``outdoor`` -- porque o corpus aqui nao e separado
por cenario e nao havia base para escolher um a priori.

Saida
-----
* ``resultados/t04_corpus_<pesos>.json`` -- metricas por split
* ``resultados/t04_escores_<pesos>.csv`` -- escore por imagem, com split e rotulo;
  e a entrada da coluna T04 na fusao T05

Uso::

    python automacao/avaliar_t04_corpus.py --conjunto tcc3_30k
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from automacao.replicar_t04_object_shadow import (      # noqa: E402
    CLASSE_PARA_INDICE, ConjuntoObjetoSombra, carregar_modelo,
)
from codigo.configuracao import EXTERNAL, RESULTS_DIR, SEED    # noqa: E402
from codigo.metricas import compute_metrics               # noqa: E402

VARIANTES = ("combined", "indoor", "outdoor")


def listar_pares(raiz: Path, conjunto: str, split: str) -> tuple[list[Path], list[Path], list[str]]:
    """Pares (sombra, objeto) de um split, na ordem dos nomes de arquivo.

    A ordenacao explicita importa: o rotulo vem do diretorio pai e o escore sera
    casado com o manifesto pelo nome do arquivo, de modo que as duas listas
    precisam estar alinhadas item a item.
    """
    base_sombra = raiz / f"{conjunto}_shadow" / split
    base_objeto = raiz / f"{conjunto}_object" / split

    if not base_sombra.exists():
        return [], [], []

    sombras, objetos, nomes = [], [], []
    for caminho in sorted(base_sombra.rglob("*.*")):
        if not caminho.is_file() or caminho.parent.name not in CLASSE_PARA_INDICE:
            continue
        par = base_objeto / caminho.parent.name / caminho.name
        if not par.exists():
            # Par incompleto so ocorre se a extracao foi interrompida entre as
            # duas gravacoes; descartar e mais seguro que inventar um mapa.
            continue
        sombras.append(caminho)
        objetos.append(par)
        nomes.append(caminho.stem)
    return sombras, objetos, nomes


def escores(modelo, sombras: list[Path], objetos: list[Path],
            dispositivo: torch.device, lote: int
            ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """P(sintetica), rotulo verdadeiro e indicador de par vazio.

    O terceiro retorno marca as amostras em que **algum** dos dois mapas saiu
    inteiramente preto -- o SSISv2 nao detectou par objeto-sombra. Ele sai
    daqui, e nao de uma segunda leitura dos arquivos, porque o ``DataLoader``
    ja tem os tensores em memoria; o custo e nulo.

    A distincao importa para a interpretacao: um par vazio nao carrega
    informacao geometrica alguma, e se a taxa de vazios diferir entre reais e
    sinteticas o classificador pode acertar por esse atalho em vez de por
    geometria. E o mesmo confundidor que ``automacao/t04_mascaras_vazias.py``
    isolou no corpus dos autores.
    """
    from torch.utils.data import DataLoader

    conjunto = ConjuntoObjetoSombra(sombras, objetos)
    carregador = DataLoader(conjunto, batch_size=lote, shuffle=False, num_workers=0)

    probabilidades, rotulos, vazios = [], [], []
    with torch.no_grad():
        for imagens, alvos in carregador:
            saidas = modelo(imagens.to(dispositivo))
            # Coluna 1 = classe "gen", como em test.py:69.
            probabilidades.append(torch.softmax(saidas, dim=1)[:, 1].cpu().numpy())
            rotulos.append(alvos.numpy())
            # imagens: (lote, 2, H, W), banda 0 = sombra, banda 1 = objeto.
            # O limiar acompanha a binariedade dos mapas; os valores 1-6 que o
            # JPEG deixa no fundo preto ficam abaixo dele.
            conteudo = (imagens > 20 / 255).flatten(2).any(dim=2)
            vazios.append((~conteudo.all(dim=1)).numpy())

    return (np.concatenate(probabilidades), np.concatenate(rotulos),
            np.concatenate(vazios))


def main() -> int:
    parser = argparse.ArgumentParser(description="T04 sobre o corpus deste trabalho")
    parser.add_argument("--raiz", type=Path, default=Path("data/t04_object_shadow"))
    parser.add_argument("--conjunto", default="tcc3_30k")
    # Sem isto, uma rodada sobre outro corpus grava por cima de
    # resultados/t04_escores_combined.csv, que e a fonte dos numeros da
    # dissertacao (AUC 0,5384). O conjunto ja separa os mapas de entrada;
    # esta opcao separa tambem a saida.
    parser.add_argument("--sufixo", default="",
                        help="sufixo no nome dos arquivos de saida, para nao "
                             "sobrescrever a rodada padrao (ex.: _ood)")
    parser.add_argument("--splits", nargs="+", default=["test", "train", "val"])
    parser.add_argument("--variantes", nargs="+", default=list(VARIANTES),
                        choices=list(VARIANTES))
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--dispositivo", choices=["auto", "cpu", "cuda"], default="auto",
                        help="'cpu' permite avaliar enquanto a GPU esta ocupada "
                             "pela extracao ou por T02")
    args = parser.parse_args()

    torch.manual_seed(SEED)
    np.random.seed(SEED)

    disponiveis = {
        split: listar_pares(args.raiz, args.conjunto, split) for split in args.splits
    }
    disponiveis = {s: v for s, v in disponiveis.items() if v[0]}
    if not disponiveis:
        print(f"ERRO: nenhum par encontrado em {args.raiz}")
        print("Rode antes a extracao: automacao/t04_extracao.bat")
        return 1

    if args.dispositivo == "auto":
        dispositivo = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        dispositivo = torch.device(args.dispositivo)
    print("=" * 72)
    print("T04 SOBRE O CORPUS DESTE TRABALHO (Corvi et al., 2024)")
    print("=" * 72)
    print(f"Mapas ........... {args.raiz / args.conjunto}_{{shadow,object}}")
    print(f"Dispositivo ..... {dispositivo}")
    for split, (sombras, _, _) in disponiveis.items():
        print(f"  {split:<6} {len(sombras)} pares")
    print("=" * 72, flush=True)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    for variante in args.variantes:
        pesos = EXTERNAL.geometry_weights / "object_shadow" / f"ShadowObject_{variante}.pth"
        if not pesos.exists():
            print(f"\n[pulado] pesos ausentes: {pesos}")
            continue

        modelo = carregar_modelo(pesos, dispositivo)
        print(f"\n--- pesos: {pesos.name} ---", flush=True)

        saida: dict = {
            "corpus": "corvi2024 (deste trabalho)",
            "pesos": str(pesos),
            "observacao": ("classificador oficial aplicado sem reajuste; "
                           "treinado em Kandinsky, avaliado em Corvi et al. (2024)"),
            "splits": {},
        }
        linhas: list[pd.DataFrame] = []

        for split, (sombras, objetos, nomes) in disponiveis.items():
            y_prob, y, vazio = escores(modelo, sombras, objetos, dispositivo,
                                       args.batch_size)
            metricas = compute_metrics(y, y_prob)

            # Estratificacao pelo confundidor das mascaras vazias.
            metricas["frac_vazios"] = float(vazio.mean())
            metricas["frac_vazios_real"] = float(vazio[y == 0].mean()) if (y == 0).any() else float("nan")
            metricas["frac_vazios_gen"] = float(vazio[y == 1].mean()) if (y == 1).any() else float("nan")

            cheio = ~vazio
            if cheio.any() and len(np.unique(y[cheio])) == 2:
                metricas["auc_pares_com_conteudo"] = float(
                    compute_metrics(y[cheio], y_prob[cheio])["auc"])
                metricas["n_pares_com_conteudo"] = int(cheio.sum())

            saida["splits"][split] = metricas

            print(f"  {split:<6} n={metricas['n_samples']:>6.0f}  "
                  f"AUC={metricas['auc']:.4f}  acc={metricas['accuracy']:.4f}  "
                  f"FPR={metricas['fpr']:.4f}", flush=True)
            print(f"         vazios {metricas['frac_vazios']:.1%} "
                  f"(reais {metricas['frac_vazios_real']:.1%}, "
                  f"sinteticas {metricas['frac_vazios_gen']:.1%})"
                  + (f"  AUC so com conteudo={metricas['auc_pares_com_conteudo']:.4f}"
                     f" (n={metricas['n_pares_com_conteudo']})"
                     if "auc_pares_com_conteudo" in metricas else ""), flush=True)

            linhas.append(pd.DataFrame({
                "arquivo": nomes, "split": split, "label": y,
                "escore_t04": y_prob, "par_vazio": vazio.astype(int),
            }))

        destino_json = RESULTS_DIR / f"t04_corpus_{variante}{args.sufixo}.json"
        destino_json.write_text(json.dumps(saida, indent=2, ensure_ascii=False),
                                encoding="utf-8")

        destino_csv = RESULTS_DIR / f"t04_escores_{variante}{args.sufixo}.csv"
        pd.concat(linhas, ignore_index=True).to_csv(destino_csv, index=False)

        print(f"  gravado: {destino_json.name}, {destino_csv.name}")

    print("\n" + "=" * 72)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
