"""Confundidor em T04: mascaras vazias como atalho de decisao.

Observacao que motivou este script
----------------------------------
As mascaras objeto-sombra publicadas pelos autores sao produzidas pelo SSISv2.
Em parte das imagens o extrator nao detecta nenhum par objeto-sombra e a
mascara sai inteiramente zerada. Amostragem inicial:

    sinteticas  24,7% de mascaras de sombra vazias
    reais       45,0%

A diferenca de 20 pontos entre as classes cria um atalho: o classificador pode
aprender "mascara vazia -> real" e acertar bastante sem analisar geometria
alguma. Seria o mesmo tipo de vies de conjunto de dados detectado na resolucao
e no formato do corpus principal (docs/DECISOES_METODOLOGICAS.md, secao 4).

Este script separa os dois efeitos, recalculando a AUC apenas sobre as imagens
em que **ambas** as mascaras tem conteudo. Se a AUC cair muito, parte do
desempenho vinha da taxa de deteccao do extrator, nao das relacoes geometricas.

Uso::

    python scripts/t04_mascaras_vazias.py --category outdoor
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import EXTERNAL, RESULTS_DIR                          # noqa: E402
from scripts.replicar_t04_object_shadow import (                      # noqa: E402
    CLASSE_PARA_INDICE, ConjuntoObjetoSombra, carregar_modelo,
    derivar_caminhos, listar_existentes,
)


def mascara_vazia(caminho: Path) -> bool:
    with Image.open(caminho) as imagem:
        return int(np.asarray(imagem).max()) == 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Efeito das mascaras vazias em T04")
    parser.add_argument("--category", default="outdoor", choices=["indoor", "outdoor"])
    parser.add_argument("--dataset-root", type=Path,
                        default=EXTERNAL.geometry_repo / "dataset")
    parser.add_argument("--batch-size", type=int, default=128)
    args = parser.parse_args()

    raiz = args.dataset_root
    pesos = EXTERNAL.geometry_weights / "object_shadow" / f"ShadowObject_{args.category}.pth"
    base = EXTERNAL.geometry_repo / "object_shadow"

    lista = (sorted(pickle.load(open(base / f"unconfident_{args.category}_list.pkl", "rb")))
             + sorted(pickle.load(open(base / f"misclassified_{args.category}_list.pkl", "rb"))))
    sombras, objetos = derivar_caminhos(lista, raiz)
    sombras, objetos, _ = listar_existentes(sombras, objetos)
    print(f"Subconjunto prequalificado: {len(sombras)} imagens\n")

    dispositivo = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    modelo = carregar_modelo(pesos, dispositivo)

    conjunto = ConjuntoObjetoSombra(sombras, objetos)
    carregador = DataLoader(conjunto, batch_size=args.batch_size, shuffle=False, num_workers=0)

    probabilidades, rotulos = [], []
    with torch.no_grad():
        for imagens, alvos in carregador:
            saidas = modelo(imagens.to(dispositivo))
            probabilidades.append(torch.softmax(saidas, dim=1)[:, 1].cpu().numpy())
            rotulos.append(alvos.numpy())
    y_prob = np.concatenate(probabilidades)
    y = np.concatenate(rotulos)

    print("Verificando conteudo das mascaras ...")
    vazias = np.array([mascara_vazia(s) or mascara_vazia(o)
                       for s, o in zip(sombras, objetos)])
    com_conteudo = ~vazias

    saida: dict = {"categoria": args.category, "n_total": int(len(y))}

    print("\n" + "=" * 70)
    print("TAXA DE MASCARAS VAZIAS POR CLASSE")
    print("=" * 70)
    for rotulo, nome in ((1, "sinteticas"), (0, "reais")):
        sel = y == rotulo
        taxa = float(vazias[sel].mean()) if sel.sum() else 0.0
        saida[f"vazias_{nome}"] = taxa
        print(f"  {nome:<12} {int(vazias[sel].sum()):>5} de {int(sel.sum()):>5}  ({taxa * 100:.1f}%)")

    diferenca = abs(saida["vazias_sinteticas"] - saida["vazias_reais"])
    saida["diferenca_entre_classes"] = diferenca
    print(f"\n  diferenca entre classes: {diferenca * 100:.1f} p.p.")

    print("\n" + "=" * 70)
    print("AUC COM E SEM AS MASCARAS VAZIAS")
    print("=" * 70)

    auc_total = float(roc_auc_score(y, y_prob))
    saida["auc_todas"] = auc_total
    print(f"  todas as imagens ............ n={len(y):>5}  AUC={auc_total:.4f}")

    if com_conteudo.sum() > 0 and len(np.unique(y[com_conteudo])) == 2:
        auc_conteudo = float(roc_auc_score(y[com_conteudo], y_prob[com_conteudo]))
        saida["auc_com_conteudo"] = auc_conteudo
        saida["n_com_conteudo"] = int(com_conteudo.sum())
        print(f"  apenas mascaras com conteudo  n={int(com_conteudo.sum()):>5}  AUC={auc_conteudo:.4f}")
        variacao = auc_conteudo - auc_total
        saida["variacao"] = variacao
        print(f"\n  variacao: {variacao * 100:+.1f} p.p.")
        if variacao < -0.03:
            print("  -> ATALHO: parte do desempenho vinha da taxa de deteccao do")
            print("     extrator, nao das relacoes geometricas. Declarar na analise.")
        elif variacao > 0.03:
            print("  -> RUIDO: as mascaras vazias nao sao atalho, sao entradas sem")
            print("     informacao que REBAIXAM a AUC. O desempenho real do")
            print("     classificador sobre geometria e o valor com conteudo.")
        else:
            print("  -> sem efeito relevante.")

    if vazias.sum() > 0 and len(np.unique(y[vazias])) == 2:
        auc_vazias = float(roc_auc_score(y[vazias], y_prob[vazias]))
        saida["auc_apenas_vazias"] = auc_vazias
        print(f"\n  apenas mascaras vazias ...... n={int(vazias.sum()):>5}  AUC={auc_vazias:.4f}")
        print("     (entrada identicamente nula; AUC diferente de 0,50 aqui e artefato)")

    destino = RESULTS_DIR / f"t04_mascaras_vazias_{args.category}.json"
    destino.write_text(json.dumps(saida, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nGravado em {destino}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
