"""Terceiro componente de T04: segmentos de reta, extraidos e classificados.

Fecha a ultima das tres representacoes geometricas de T04. Com esta, as tres
passam a ter extrator funcionando no ambiente WSL2 descrito em
``documentacao/T04_AMBIENTE_WSL2.md``.

O extrator
----------
DeepLSD (Pautrat et al., 2023), que o artigo de Sarkar et al. (2024) **nomeia**
explicitamente: *"identifying key structural lines using Deep LSD"*. Diferente do
PerspectiveFields, onde os autores dizem apenas "a pretrained model", aqui nao
ha duvida sobre qual detector usar.

Formato, verificado contra o arquivo dos autores
------------------------------------------------
Os autores distribuem ``image_path_to_lines.pkl`` com as retas ja detectadas do
corpus deles. Inspecionando-o::

    forma      (N, 4) float32     -> x1, y1, x2, y2
    intervalo  -1,1 a 256,4       -> pixels crus, quadro 256x256
    contagem   mediana 81 retas por imagem

A saida do DeepLSD sobre o corpus deste trabalho::

    forma      (N, 2, 2) -> achatada em (N, 4)
    intervalo  0,8 a 256,6
    contagem   73 retas na primeira imagem

Coincide em forma, escala e ordem de grandeza. E o quadro de 256x256 e o mesmo
do corpus normalizado aqui, entao nao ha reescala a fazer -- alimentar o
classificador com pixels em outro quadro produziria numeros sem sentido.

As retas sao gravadas em disco, no mesmo formato dos autores. Sao poucos KB por
imagem (ao contrario dos 768 KB dos campos de perspectiva), o que torna o
artefato barato e auditavel.

O PointNet oficial nao roda em CPU se houver GPU
------------------------------------------------
``lines_model.py:39-41`` move a matriz identidade para CUDA sempre que CUDA
esta **disponivel**, e nao quando o modelo esta na GPU::

    identity_matrix = torch.eye(self.output_dim)
    if torch.cuda.is_available():
        identity_matrix = identity_matrix.cuda()
    x = x.view(...) + identity_matrix

Numa maquina com GPU, pedir CPU produz ``Expected all tensors to be on the same
device``. Nao e problema em producao -- a extracao roda em GPU de qualquer modo
--, mas impede testar em CPU enquanto a placa esta ocupada por outra execucao.

Contorno, sem alterar o codigo dos autores: esconder a placa com
``CUDA_VISIBLE_DEVICES=""``. Ai ``torch.cuda.is_available()`` devolve falso e o
caminho de CPU funciona.

Incerteza registrada
--------------------
Os autores nao informam os parametros de deteccao nem qual dos dois checkpoints
do DeepLSD usaram (wireframe, para interiores, ou MegaDepth, generico). Usa-se
``deeplsd_md.tar`` com os parametros do exemplo oficial. Como em
``extrair_perspective_fields.py``, a escolha esta declarada e limita a
comparacao com o valor publicado.

Uso (dentro do WSL2)::

    /root/geo/bin/python extrair_line_segments.py \
        --manifesto .../manifesto30k_standard.csv --split test --variante combined
"""

from __future__ import annotations

import argparse
import csv
import json
import pickle
import time
from pathlib import Path

import numpy as np

ROTULO_PARA_CLASSE = {0: "real", 1: "gen"}
N_RETAS = 250          # lines_dataset.py:9

# Parametros do exemplo oficial do DeepLSD. Nao ha registro de quais Sarkar
# et al. usaram.
CONF_DETECCAO = {
    "detect_lines": True,
    "line_detection_params": {
        "merge": False,
        "filtering": True,
        "grad_thresh": 3,
        "grad_nfa": True,
    },
}


def caminho_windows_para_wsl(bruto: str) -> Path:
    texto = bruto.strip().replace("\\", "/")
    if len(texto) > 1 and texto[1] == ":":
        return Path(f"/mnt/{texto[0].lower()}{texto[2:]}")
    return Path(texto)


def amostrar(retas: np.ndarray, gerador: np.random.Generator) -> np.ndarray:
    """Fixa 250 retas, como ``LineSegmentDataset.__getitem__``.

    Nos dados dos autores este passo e determinista na pratica: nenhuma imagem
    do subconjunto prequalificado chega a 250 retas, entao so o ramo de
    duplicacao roda, e o max-pooling do PointNet e insensivel a duplicatas. Aqui
    o mesmo tende a valer -- as contagens sao da mesma ordem.
    """
    faltam = N_RETAS - retas.shape[0]
    if faltam > 0:
        indices = gerador.choice(retas.shape[0], faltam)
        return np.concatenate((retas, retas[indices, :]), axis=0)
    return retas[gerador.choice(retas.shape[0], N_RETAS), :]


def carregar_classificador(pesos: Path, arquitetura: Path, dispositivo):
    """PointNet oficial (``lines_model.py``), carregado por caminho."""
    import importlib.util

    import torch

    espec = importlib.util.spec_from_file_location("lines_model", arquitetura)
    modulo = importlib.util.module_from_spec(espec)
    espec.loader.exec_module(modulo)

    modelo = modulo.ClassificationPointNet(num_classes=2, point_dimension=4)
    modelo.load_state_dict(torch.load(pesos, map_location=dispositivo))
    modelo.to(dispositivo)
    modelo.eval()
    return modelo


def carregar_detector(checkpoint: Path, dispositivo):
    import torch
    from deeplsd.models.deeplsd_inference import DeepLSD

    estado = torch.load(checkpoint, map_location="cpu")
    rede = DeepLSD(CONF_DETECCAO)
    rede.load_state_dict(estado["model"])
    return rede.eval().to(dispositivo)


def ler_entradas(manifesto: Path, split: str | None, limite: int) -> list[dict]:
    entradas: list[dict] = []
    with open(manifesto, newline="", encoding="utf-8") as arquivo:
        for linha in csv.DictReader(arquivo):
            if split and linha.get("split") != split:
                continue
            origem = caminho_windows_para_wsl(linha["path"])
            entradas.append({
                "origem": origem, "nome": origem.stem,
                "divisao": linha.get("split") or "all",
                "rotulo": int(linha["label"]),
                "gerador": linha.get("generator", ""),
            })
    return entradas[:limite] if limite else entradas


def main() -> int:
    parser = argparse.ArgumentParser(description="T04, componente de segmentos de reta")
    parser.add_argument("--manifesto", type=Path, required=True)
    parser.add_argument("--split", default=None)
    parser.add_argument("--saida", type=Path, required=True)
    parser.add_argument("--conjunto", default="tcc3_30k")
    parser.add_argument("--variante", default="combined",
                        choices=["combined", "indoor", "outdoor"])
    parser.add_argument("--raiz-projeto", type=Path,
                        default=Path("/mnt/c/Users/ferre/projects/tcc3-deteccao-imagens-ia"))
    parser.add_argument("--checkpoint", type=Path,
                        default=Path("/root/DeepLSD/weights/deeplsd_md.tar"))
    parser.add_argument("--dispositivo", default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--limite", type=int, default=0)
    parser.add_argument("--semente", type=int, default=42)
    args = parser.parse_args()

    pesos = (args.raiz_projeto / "pesos" / "projective_geometry" /
             "line_segment" / f"Lines_{args.variante}.pt")
    arquitetura = (args.raiz_projeto / "externo" / "projective-geometry" /
                   "line_segment" / "lines_model.py")

    for caminho in (args.manifesto, pesos, arquitetura, args.checkpoint):
        if not caminho.exists():
            print(f"ERRO: nao encontrado: {caminho}")
            return 1

    entradas = ler_entradas(args.manifesto, args.split, args.limite)
    if not entradas:
        print("ERRO: nenhuma entrada apos o filtro de split")
        return 1

    args.saida.mkdir(parents=True, exist_ok=True)
    sufixo = f"_{args.split}" if args.split else ""
    destino_csv = args.saida / f"{args.conjunto}_ls_{args.variante}{sufixo}.csv"
    destino_retas = args.saida / f"{args.conjunto}_retas{sufixo}.pkl"

    if destino_csv.exists():
        print(f"[retomada] {destino_csv.name} ja existe; nada a fazer")
        return 0

    print("=" * 72)
    print("T04 - SEGMENTOS DE RETA (extracao + classificacao)")
    print("=" * 72)
    print(f"Manifesto ....... {args.manifesto}")
    print(f"Split ........... {args.split or 'todos'}")
    print(f"Imagens ......... {len(entradas)}")
    print(f"Detector ........ {args.checkpoint.name}")
    print(f"Classificador ... {pesos.name}")
    print(f"Dispositivo ..... {args.dispositivo}")
    print("=" * 72, flush=True)

    import cv2
    import torch

    detector = carregar_detector(args.checkpoint, args.dispositivo)
    classificador = carregar_classificador(pesos, arquitetura,
                                           torch.device(args.dispositivo))
    print("Modelos carregados.\n", flush=True)

    gerador = np.random.default_rng(args.semente)
    registro: list[dict] = []
    retas_por_imagem: dict[str, np.ndarray] = {}
    falhas: list[dict] = []
    inicio = time.perf_counter()

    for indice, item in enumerate(entradas, start=1):
        imagem = cv2.imread(str(item["origem"]))
        if imagem is None:
            falhas.append({"arquivo": item["nome"], "erro": "cv2.imread devolveu None"})
            continue

        try:
            cinza = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)
            entrada = {"image": torch.tensor(cinza, dtype=torch.float,
                                             device=args.dispositivo)[None, None] / 255.0}
            with torch.no_grad():
                retas = np.asarray(detector(entrada)["lines"][0], dtype=np.float32)
            if retas.size == 0:
                # Imagem sem nenhuma reta detectada: nao ha o que classificar.
                falhas.append({"arquivo": item["nome"], "erro": "nenhuma reta detectada"})
                continue
            retas = retas.reshape(retas.shape[0], -1)      # (N, 2, 2) -> (N, 4)

            with torch.no_grad():
                lote = torch.from_numpy(amostrar(retas, gerador)[None]).to(args.dispositivo)
                saida = classificador(lote)
                escore = float(torch.softmax(saida, dim=1)[0, 1])
        except Exception as erro:                       # noqa: BLE001
            print(f"[falha] {type(erro).__name__} em {item['nome']}: {erro}", flush=True)
            falhas.append({"arquivo": item["nome"],
                           "erro": f"{type(erro).__name__}: {erro}"})
            continue

        retas_por_imagem[item["nome"]] = retas
        registro.append({
            "arquivo": item["nome"], "split": item["divisao"],
            "label": item["rotulo"], "gerador": item["gerador"],
            "n_retas": int(retas.shape[0]), "escore_t04_ls": escore,
        })

        if len(registro) % 250 == 0:
            decorrido = time.perf_counter() - inicio
            taxa = decorrido / len(registro)
            print(f"  {indice}/{len(entradas)}  {taxa:.3f} s/img  "
                  f"restam ~{(len(entradas) - indice) * taxa / 60:.0f} min", flush=True)

    decorrido = time.perf_counter() - inicio
    print("\n" + "=" * 72)
    print(f"Processadas ..... {len(registro)}")
    print(f"Falhas .......... {len(falhas)}")

    if registro:
        print(f"Tempo ........... {decorrido / 60:.1f} min "
              f"({decorrido / len(registro):.3f} s/img)")
        contagens = np.array([r["n_retas"] for r in registro])
        print(f"Retas por imagem  mediana {int(np.median(contagens))}, "
              f"max {contagens.max()}, acima de {N_RETAS}: {(contagens > N_RETAS).sum()}")

        with open(destino_csv, "w", newline="", encoding="utf-8") as arquivo:
            escritor = csv.DictWriter(arquivo, fieldnames=list(registro[0].keys()))
            escritor.writeheader()
            escritor.writerows(registro)
        with open(destino_retas, "wb") as arquivo:
            pickle.dump(retas_por_imagem, arquivo)
        print(f"\nEscores em {destino_csv}")
        print(f"Retas em   {destino_retas}")

        escores = np.array([r["escore_t04_ls"] for r in registro])
        rotulos = np.array([r["label"] for r in registro])
        if len(np.unique(rotulos)) == 2:
            from sklearn.metrics import roc_auc_score
            print(f"AUC parcial deste split: {roc_auc_score(rotulos, escores):.4f}")

    if falhas:
        destino_falhas = args.saida / f"{args.conjunto}_ls_falhas{sufixo}.csv"
        with open(destino_falhas, "w", newline="", encoding="utf-8") as arquivo:
            escritor = csv.DictWriter(arquivo, fieldnames=["arquivo", "erro"])
            escritor.writeheader()
            escritor.writerows(falhas)
        print(f"Falhas em  {destino_falhas}")

    (args.saida / f"{args.conjunto}_ls_{args.variante}{sufixo}.json").write_text(
        json.dumps({
            "manifesto": str(args.manifesto), "split": args.split,
            "detector": str(args.checkpoint), "conf_deteccao": CONF_DETECCAO,
            "classificador": str(pesos), "n_processadas": len(registro),
            "n_falhas": len(falhas),
            "segundos_por_imagem": decorrido / len(registro) if registro else None,
        }, indent=2), encoding="utf-8")
    print("=" * 72)
    return 0 if not falhas else 1


if __name__ == "__main__":
    raise SystemExit(main())
