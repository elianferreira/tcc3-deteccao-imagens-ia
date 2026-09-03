"""Replicacao parcial de T04: classificador objeto-sombra (Sarkar et al., 2024).

Contexto
--------
T04 nao pode ser integrada as demais tecnicas: seus tres classificadores nao
recebem pixels, e sim representacoes geometricas ja extraidas por modelos
auxiliares dependentes de detectron2, sem suporte em Windows. Ver
``externo/CONTRATO.md`` e ``documentacao/DECISOES_METODOLOGICAS.md``, secao 5.

Uma replicacao parcial e possivel porque os autores publicaram, em um segundo
repositorio (``Projective-Geometry-OS``), as mascaras objeto-sombra ja
extraidas para o proprio conjunto de teste. Este script avalia o classificador
oficial sobre essas mascaras.

O que este resultado e e o que nao e
------------------------------------
E: uma verificacao de que o classificador oficial de objeto-sombra reproduz o
desempenho publicado, executado com os pesos oficiais.

NAO e: um resultado comparavel a T01, T02, T03 e T05. O corpus e outro
(Kandinsky, dos autores) e nao o de Corvi et al. (2024); a metrica nao entra na
tabela comparativa nem na fusao T05, e a tecnica permanece indisponivel na
interface.

Por que um adaptador em vez de ``object_shadow/test.py``
--------------------------------------------------------
O script oficial divide caminhos com ``split("/")`` em quatro pontos
(``test.py`` 19, 23, 81 e ``dataset.py`` 28). No Windows ``glob`` devolve
caminhos com barra invertida, de modo que ``dataset.py:28`` produziria a string
inteira em vez do nome da classe, e o rotulo falharia. Reimplementar a leitura
aqui preserva o repositorio oficial intacto e mantem a replicacao auditavel: a
arquitetura, os pesos e a composicao dos subconjuntos seguem os originais.

Subconjuntos
------------
O ``test.py`` oficial deixa o subconjunto ``easy`` comentado (linha 130) e
reporta apenas ``unconfident`` e ``misclassified`` -- as imagens que enganam
detectores baseados em sinal, que sao o alvo do artigo. Este script avalia os
tres e identifica cada um, para que a comparacao com o valor publicado seja
feita sobre o subconjunto correto.

Uso::

    python automacao/replicar_t04_object_shadow.py --category outdoor
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torchvision
from PIL import Image
from sklearn.metrics import confusion_matrix, roc_auc_score
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from codigo.configuracao import EXTERNAL, RESULTS_DIR, SEED      # noqa: E402

# Rotulos do trabalho original (test.py:142): 0 = real, 1 = gerada.
CLASSE_PARA_INDICE = {"real": 0, "gen": 1}


class ConjuntoObjetoSombra(Dataset):
    """Le mascara de sombra e de objeto e as empilha em duas bandas.

    Equivalente a ``ShadowObjectDataset`` oficial (``object_shadow/dataset.py``),
    porem com extracao de rotulo independente de separador de caminho.
    """

    def __init__(self, caminhos_sombra: list[Path], caminhos_objeto: list[Path]):
        self.caminhos_sombra = caminhos_sombra
        self.caminhos_objeto = caminhos_objeto
        self.para_tensor = torchvision.transforms.ToTensor()

    def __len__(self) -> int:
        return len(self.caminhos_sombra)

    def __getitem__(self, indice: int):
        caminho_sombra = self.caminhos_sombra[indice]
        caminho_objeto = self.caminhos_objeto[indice]

        sombra = self.para_tensor(Image.open(caminho_sombra).convert("L"))
        objeto = self.para_tensor(Image.open(caminho_objeto).convert("L"))
        empilhado = torch.cat([sombra, objeto], dim=0)

        # O rotulo e o nome do diretorio pai: .../test/<classe>/<arquivo>
        rotulo = CLASSE_PARA_INDICE[caminho_sombra.parent.name]
        return empilhado, rotulo


def derivar_caminhos(lista_imagens: list[str], raiz: Path) -> tuple[list[Path], list[Path]]:
    """Converte caminhos de imagem nos caminhos de mascara correspondentes.

    Reproduz ``get_shadow_object_paths`` (test.py:15-26): o nome do diretorio do
    conjunto recebe os sufixos ``_shadow`` e ``_object``. Entradas vem dos
    pickles oficiais no formato ``../dataset/Kandinsky_Outdoor/test/gen/x.jpg``.
    """
    sombras, objetos = [], []
    for bruto in lista_imagens:
        partes = bruto.replace("\\", "/").split("/")
        # partes[-4] e o nome do conjunto; -3 = "test", -2 = classe, -1 = arquivo
        conjunto, divisao, classe, arquivo = partes[-4], partes[-3], partes[-2], partes[-1]
        sombras.append(raiz / f"{conjunto}_shadow" / divisao / classe / arquivo)
        objetos.append(raiz / f"{conjunto}_object" / divisao / classe / arquivo)
    return sombras, objetos


def listar_existentes(sombras: list[Path], objetos: list[Path]) -> tuple[list[Path], list[Path], int]:
    """Mantem apenas os pares cujas duas mascaras existem em disco."""
    s_ok, o_ok = [], []
    for sombra, objeto in zip(sombras, objetos):
        if sombra.exists() and objeto.exists():
            s_ok.append(sombra)
            o_ok.append(objeto)
    return s_ok, o_ok, len(sombras) - len(s_ok)


def carregar_modelo(caminho_pesos: Path, dispositivo: torch.device) -> nn.Module:
    """ResNet-50 com primeira convolucao de 2 bandas (test.py:112-114)."""
    modelo = torchvision.models.resnet50(weights=None)
    modelo.conv1 = nn.Conv2d(2, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
    modelo.fc = nn.Linear(in_features=2048, out_features=2, bias=True)

    estado = torch.load(caminho_pesos, map_location=dispositivo)
    modelo.load_state_dict(estado)
    modelo.to(dispositivo)
    modelo.eval()
    return modelo


def avaliar(modelo: nn.Module, conjunto: Dataset, dispositivo: torch.device,
            lote: int = 128) -> dict:
    if len(conjunto) == 0:
        return {"n": 0}

    carregador = DataLoader(conjunto, batch_size=lote, shuffle=False, num_workers=0)
    probabilidades, rotulos = [], []

    with torch.no_grad():
        for imagens, alvos in carregador:
            saidas = modelo(imagens.to(dispositivo))
            # Coluna 1 = classe "gen", como em test.py:69.
            probabilidades.append(torch.softmax(saidas, dim=1)[:, 1].cpu().numpy())
            rotulos.append(alvos.numpy())

    y_prob = np.concatenate(probabilidades)
    y = np.concatenate(rotulos)
    y_pred = (y_prob >= 0.5).astype(int)

    resultado = {"n": int(len(y)), "n_gen": int((y == 1).sum()), "n_real": int((y == 0).sum())}
    if len(np.unique(y)) < 2:
        resultado["auc"] = float("nan")
        resultado["observacao"] = "subconjunto com uma unica classe; AUC indefinida"
    else:
        resultado["auc"] = float(roc_auc_score(y, y_prob))

    resultado["acuracia"] = float((y_pred == y).mean())
    matriz = confusion_matrix(y, y_pred, labels=[0, 1])
    resultado["matriz_confusao"] = matriz.tolist()
    return resultado


def main() -> int:
    parser = argparse.ArgumentParser(description="Replicacao parcial de T04 (objeto-sombra)")
    parser.add_argument("--category", default="outdoor", choices=["indoor", "outdoor"])
    parser.add_argument("--dataset-root", type=Path,
                        default=EXTERNAL.geometry_repo / "dataset")
    parser.add_argument("--batch-size", type=int, default=128)
    args = parser.parse_args()

    torch.manual_seed(SEED)
    np.random.seed(SEED)

    raiz = args.dataset_root
    if not raiz.exists():
        print(f"ERRO: dataset nao encontrado em {raiz}")
        return 1

    pesos = EXTERNAL.geometry_weights / "object_shadow" / f"ShadowObject_{args.category}.pth"
    if not pesos.exists():
        print(f"ERRO: pesos oficiais ausentes em {pesos}")
        return 1

    base = EXTERNAL.geometry_repo / "object_shadow"
    lista_mal = sorted(pickle.load(open(base / f"misclassified_{args.category}_list.pkl", "rb")))
    lista_incerta = sorted(pickle.load(open(base / f"unconfident_{args.category}_list.pkl", "rb")))

    dispositivo = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Categoria ......... {args.category}")
    print(f"Dispositivo ....... {dispositivo}")
    print(f"Pesos oficiais .... {pesos.name}")
    print(f"Dataset ........... {raiz}\n")

    modelo = carregar_modelo(pesos, dispositivo)
    print("Modelo oficial carregado.\n")

    # Todos os pares presentes no disco, para compor o subconjunto "easy".
    conjunto_nome = "Kandinsky_Indoor" if args.category == "indoor" else "Kandinsky_Outdoor"
    todas_sombras = sorted((raiz / f"{conjunto_nome}_shadow" / "test").rglob("*.*"))
    todas = [
        p for p in todas_sombras
        if p.parent.name in CLASSE_PARA_INDICE and p.is_file()
    ]

    subconjuntos: dict[str, tuple[list[Path], list[Path]]] = {}

    for nome, lista in (("misclassified", lista_mal), ("unconfident", lista_incerta)):
        sombras, objetos = derivar_caminhos(lista, raiz)
        sombras, objetos, ausentes = listar_existentes(sombras, objetos)
        if ausentes:
            print(f"[aviso] {nome}: {ausentes} pares sem mascara em disco, descartados")
        subconjuntos[nome] = (sombras, objetos)

    dificeis = {p.resolve() for p in subconjuntos["misclassified"][0] + subconjuntos["unconfident"][0]}
    faceis_sombra = [p for p in todas if p.resolve() not in dificeis]
    faceis_objeto = [
        raiz / f"{conjunto_nome}_object" / p.parent.parent.name / p.parent.name / p.name
        for p in faceis_sombra
    ]
    faceis_sombra, faceis_objeto, _ = listar_existentes(faceis_sombra, faceis_objeto)
    subconjuntos["easy"] = (faceis_sombra, faceis_objeto)

    saida: dict = {
        "categoria": args.category,
        "corpus": f"{conjunto_nome} (Sarkar et al., 2024)",
        "pesos": str(pesos),
        "subconjuntos": {},
    }

    print("=" * 70)
    print("RESULTADOS POR SUBCONJUNTO")
    print("=" * 70)
    for nome in ("easy", "unconfident", "misclassified"):
        sombras, objetos = subconjuntos[nome]
        resultado = avaliar(modelo, ConjuntoObjetoSombra(sombras, objetos),
                            dispositivo, args.batch_size)
        saida["subconjuntos"][nome] = resultado
        if resultado["n"] == 0:
            print(f"  {nome:<14} vazio")
            continue
        print(f"  {nome:<14} n={resultado['n']:>5}  "
              f"AUC={resultado['auc']:.4f}  acuracia={resultado['acuracia']:.4f}")

    # Uniao de unconfident e misclassified: o regime "prequalificado" do artigo.
    s_dif = subconjuntos["unconfident"][0] + subconjuntos["misclassified"][0]
    o_dif = subconjuntos["unconfident"][1] + subconjuntos["misclassified"][1]
    resultado = avaliar(modelo, ConjuntoObjetoSombra(s_dif, o_dif), dispositivo, args.batch_size)
    saida["subconjuntos"]["prequalificado"] = resultado
    if resultado["n"]:
        print(f"  {'prequalificado':<14} n={resultado['n']:>5}  "
              f"AUC={resultado['auc']:.4f}  acuracia={resultado['acuracia']:.4f}")
    print("=" * 70)

    destino = RESULTS_DIR / f"t04_object_shadow_{args.category}.json"
    destino.parent.mkdir(parents=True, exist_ok=True)
    destino.write_text(json.dumps(saida, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nGravado em {destino}")

    linhas = [
        {"tecnica": "T04", "componente": "object_shadow", "categoria": args.category,
         "subconjunto": nome, "n": r.get("n", 0), "auc": r.get("auc"),
         "acuracia": r.get("acuracia")}
        for nome, r in saida["subconjuntos"].items()
    ]
    csv_destino = RESULTS_DIR / f"t04_object_shadow_{args.category}.csv"
    pd.DataFrame(linhas).to_csv(csv_destino, index=False)
    print(f"Gravado em {csv_destino}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
