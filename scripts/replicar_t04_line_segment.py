"""Replicacao do terceiro classificador de T04: segmentos de reta.

Terceira e ultima das representacoes geometricas de T04. Este script faz por
ela o que ``replicar_t04_object_shadow.py`` faz pela primeira: avalia o
classificador **oficial** sobre os dados **dos autores**, verificando que os
pesos reproduzem o desempenho publicado.

Nao e um resultado comparavel a T01, T02, T03 e T05 -- o corpus e o Kandinsky
dos autores. Serve de verificacao da replicacao e de linha de base para a
extracao sobre o corpus deste trabalho.

Por que foi possivel sem extrator
---------------------------------
Os autores distribuem ``image_path_to_lines.pkl`` (1,8 GB, 1.049.919 imagens)
junto do dataset de segmentos de reta. Ele traz as retas ja detectadas, do mesmo
modo que ``Projective-Geometry-OS`` traz as mascaras objeto-sombra prontas.

A inspecao desse arquivo resolveu a duvida que bloqueava a replicacao: **qual a
convencao das coordenadas**, que nenhum ponto do codigo oficial documenta.

    shape        (N, 4) float32       -> x1, y1, x2, y2, extremos achatados
    intervalo    -1,1 a 256,4         -> pixels crus em quadro 256x256

Sao pixels, nao coordenadas normalizadas, e o quadro e 256x256 -- o mesmo do
corpus normalizado deste trabalho. Isso elimina a ambiguidade de escala que
tornaria a extracao propria arriscada: alimentar o classificador com pixels em
outro quadro produziria numeros sem sentido, e nao havia como saber qual usar
sem este arquivo.

Amostragem estocastica
----------------------
``LineSegmentDataset`` fixa 250 retas por imagem: reamostra com reposicao quando
ha menos, subamostra quando ha mais (``lines_dataset.py:24-32``). As duas usam
``np.random.choice``, de modo que a metrica varia entre execucoes. O script
oficial nao fixa semente; aqui ela e fixada em ``SEED`` para que o numero seja
reproduzivel, e a variacao entre sementes e reportada com ``--repeticoes``.

Uso::

    python scripts/replicar_t04_line_segment.py --category outdoor
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import pickle
import statistics
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import confusion_matrix, roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import DATA_DIR, EXTERNAL, RESULTS_DIR, SEED      # noqa: E402

CLASSE_PARA_INDICE = {"real": 0, "gen": 1}
N_RETAS = 250          # lines_dataset.py:9, valor padrao dos autores


def carregar_arquitetura():
    """Importa ``ClassificationPointNet`` do repositorio oficial, sem copiar.

    O modulo so depende de ``torch``, entao pode ser carregado por caminho sem
    arrastar o resto do repositorio -- e a arquitetura permanece a original, o
    que mantem a replicacao auditavel.
    """
    caminho = EXTERNAL.geometry_repo / "line_segment" / "lines_model.py"
    if not caminho.exists():
        raise FileNotFoundError(f"modulo oficial ausente: {caminho}")
    espec = importlib.util.spec_from_file_location("lines_model", caminho)
    modulo = importlib.util.module_from_spec(espec)
    espec.loader.exec_module(modulo)
    return modulo.ClassificationPointNet


def carregar_modelo(pesos: Path, dispositivo: torch.device):
    """PointNet de duas classes sobre pontos de dimensao 4 (lines_model.py:116)."""
    ClassificationPointNet = carregar_arquitetura()
    modelo = ClassificationPointNet(num_classes=2, point_dimension=4)
    estado = torch.load(pesos, map_location=dispositivo)
    modelo.load_state_dict(estado)
    modelo.to(dispositivo)
    modelo.eval()
    return modelo


def amostrar(retas: np.ndarray, gerador: np.random.Generator) -> np.ndarray:
    """Fixa o numero de retas em 250, como ``LineSegmentDataset.__getitem__``."""
    faltam = N_RETAS - retas.shape[0]
    if faltam > 0:
        indices = gerador.choice(retas.shape[0], faltam)
        return np.concatenate((retas, retas[indices, :]), axis=0)
    indices = gerador.choice(retas.shape[0], N_RETAS)
    return retas[indices, :]


def avaliar(modelo, caminhos: list[str], mapa: dict, dispositivo: torch.device,
            semente: int, lote: int = 256) -> dict:
    if not caminhos:
        return {"n": 0}

    gerador = np.random.default_rng(semente)
    probabilidades, rotulos = [], []

    with torch.no_grad():
        for inicio in range(0, len(caminhos), lote):
            bloco = caminhos[inicio:inicio + lote]
            pilha, alvos = [], []
            for caminho in bloco:
                retas = np.asarray(mapa[caminho], dtype=np.float32)
                if retas.size == 0:
                    continue
                pilha.append(amostrar(retas, gerador))
                alvos.append(CLASSE_PARA_INDICE[caminho.replace("\\", "/").split("/")[-2]])
            if not pilha:
                continue
            entrada = torch.from_numpy(np.stack(pilha)).to(dispositivo)
            saida = modelo(entrada)
            probabilidades.append(torch.softmax(saida, dim=1)[:, 1].cpu().numpy())
            rotulos.append(np.asarray(alvos))

    y_prob = np.concatenate(probabilidades)
    y = np.concatenate(rotulos)
    y_pred = (y_prob >= 0.5).astype(int)

    resultado = {"n": int(len(y)), "n_gen": int((y == 1).sum()),
                 "n_real": int((y == 0).sum()),
                 "acuracia": float((y_pred == y).mean())}
    if len(np.unique(y)) < 2:
        resultado["auc"] = float("nan")
        resultado["observacao"] = "subconjunto com uma unica classe"
    else:
        resultado["auc"] = float(roc_auc_score(y, y_prob))
    resultado["matriz_confusao"] = confusion_matrix(y, y_pred, labels=[0, 1]).tolist()
    return resultado


def main() -> int:
    parser = argparse.ArgumentParser(description="Replicacao de T04, componente de retas")
    parser.add_argument("--category", default="outdoor",
                        choices=["indoor", "outdoor", "combined"])
    parser.add_argument("--lines", type=Path,
                        default=DATA_DIR / "projective_geometry_lines" / "image_path_to_lines.pkl")
    parser.add_argument("--repeticoes", type=int, default=3,
                        help="sementes distintas, para medir a variacao da amostragem")
    parser.add_argument("--batch-size", type=int, default=256)
    args = parser.parse_args()

    pesos = EXTERNAL.geometry_weights / "line_segment" / f"Lines_{args.category}.pt"
    base = EXTERNAL.geometry_repo / "line_segment"
    for caminho in (args.lines, pesos):
        if not caminho.exists():
            print(f"ERRO: nao encontrado: {caminho}")
            return 1

    dispositivo = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 74)
    print("T04 - REPLICACAO DO CLASSIFICADOR DE SEGMENTOS DE RETA")
    print("=" * 74)
    print(f"Categoria ....... {args.category}")
    print(f"Dispositivo ..... {dispositivo}")
    print(f"Pesos oficiais .. {pesos.name}")
    print(f"Retas ........... {args.lines.name}")
    print("=" * 74, flush=True)

    print("Carregando as retas (1,8 GB) ...", flush=True)
    with args.lines.open("rb") as arquivo:
        mapa = pickle.load(arquivo)
    print(f"  {len(mapa)} imagens no mapa\n", flush=True)

    modelo = carregar_modelo(pesos, dispositivo)

    conjunto = {"indoor": "Kandinsky_Indoor", "outdoor": "Kandinsky_Outdoor",
                "combined": None}[args.category]

    lista_mal = sorted(pickle.load(open(base / f"misclassified_{args.category}_list.pkl", "rb")))
    lista_incerta = sorted(pickle.load(open(base / f"unconfident_{args.category}_list.pkl", "rb")))

    presente = set(mapa)
    dificeis = set(lista_mal) | set(lista_incerta)
    faceis = [
        c for c in presente
        if "/test/" in c
        and c.replace("\\", "/").split("/")[-2] in CLASSE_PARA_INDICE
        and c not in dificeis
        and (conjunto is None or f"/{conjunto}/" in c)
    ]

    subconjuntos = {
        "easy": sorted(faceis),
        "unconfident": [c for c in lista_incerta if c in presente],
        "misclassified": [c for c in lista_mal if c in presente],
    }
    subconjuntos["prequalificado"] = (subconjuntos["unconfident"]
                                      + subconjuntos["misclassified"])

    saida: dict = {
        "categoria": args.category,
        "corpus": f"{conjunto or 'Kandinsky indoor + outdoor'} (Sarkar et al., 2024)",
        "pesos": str(pesos),
        "retas": str(args.lines),
        "n_retas_por_imagem": N_RETAS,
        "subconjuntos": {},
    }

    print("=" * 74)
    print("RESULTADOS POR SUBCONJUNTO")
    print("=" * 74)
    for nome in ("easy", "unconfident", "misclassified", "prequalificado"):
        caminhos = subconjuntos[nome]
        if not caminhos:
            print(f"  {nome:<16} vazio")
            continue

        aucs = []
        for repeticao in range(args.repeticoes):
            resultado = avaliar(modelo, caminhos, mapa, dispositivo,
                                SEED + repeticao, args.batch_size)
            aucs.append(resultado["auc"])
        resultado["auc_media"] = float(statistics.mean(aucs))
        resultado["auc_desvio"] = float(statistics.stdev(aucs)) if len(aucs) > 1 else 0.0
        saida["subconjuntos"][nome] = resultado

        print(f"  {nome:<16} n={resultado['n']:>6}  "
              f"AUC={resultado['auc_media']:.4f} +/- {resultado['auc_desvio']:.4f}  "
              f"acuracia={resultado['acuracia']:.4f}")
    print("=" * 74)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    destino = RESULTS_DIR / f"t04_line_segment_{args.category}.json"
    destino.write_text(json.dumps(saida, indent=2, ensure_ascii=False), encoding="utf-8")

    linhas = [{"tecnica": "T04", "componente": "line_segment",
               "categoria": args.category, "subconjunto": nome,
               "n": r.get("n", 0), "auc": r.get("auc_media"),
               "auc_desvio": r.get("auc_desvio"), "acuracia": r.get("acuracia")}
              for nome, r in saida["subconjuntos"].items()]
    pd.DataFrame(linhas).to_csv(
        RESULTS_DIR / f"t04_line_segment_{args.category}.csv", index=False)
    print(f"\nGravado em {destino.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
