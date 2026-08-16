"""Extrai os mapas objeto-sombra de T04 com o SSISv2, dentro do WSL2.

Este script fecha a lacuna descrita em ``docs/T04_AMBIENTE_WSL2.md``: o SSISv2
ja roda na GPU, mas devolve ``N`` mascaras de instancia, enquanto o
classificador oficial de T04 consome **duas** imagens em tons de cinza por
amostra -- uma de sombra, uma de objeto (``object_shadow/dataset.py:20-21``).
O agregador esta em :func:`agregar_por_classe`.

Onde roda
---------
Somente dentro do WSL2, com o interpretador que tem detectron2 compilado com
CUDA::

    wsl -d Ubuntu-24.04 -u root -- /root/geo/bin/python \
        /mnt/c/.../scripts/wsl/extrair_object_shadow_ssis.py --manifesto ... --split test

O ambiente e os seis obstaculos vencidos para monta-lo estao registrados em
``docs/T04_AMBIENTE_WSL2.md``. Rodar fora do WSL falha na importacao do
detectron2, e rodar em CPU falha em ``Deformable Conv is not supported on
CPUs!``.

Formato de saida, e por que exatamente este
-------------------------------------------
Os mapas publicados pelos autores (``Kandinsky_Outdoor_{shadow,object}``) foram
inspecionados antes de escrever este agregador, porque o classificador foi
treinado neles e a representacao precisa coincidir. Amostra de 200 arquivos por
classe:

* modo ``L``, 256x256, uint8;
* estritamente binarios -- **zero** massa no intervalo [20, 235]; os valores 1-6
  e 249-253 que aparecem sao artefato de JPEG, nao codificacao de instancia;
* gravados em JPEG.

Disso decorrem tres decisoes, todas verificaveis contra aquela amostra:

1. as ``N`` instancias de uma classe entram por **uniao binaria**, sem
   distinguir instancia -- nao ha nivel de cinza reservado a isso;
2. a saida e 256x256, modo ``L``, valores em {0, 255};
3. o formato padrao e JPEG, para reproduzir a distribuicao em que o
   classificador foi treinado, incluindo o ringing de borda. ``--formato png``
   grava sem perda, para quem preferir o mapa exato.

Uma imagem sem nenhuma instancia detectada produz um par de mapas
integralmente pretos -- que e o que os autores tambem produzem (ha exemplos no
proprio conjunto deles). Esse caso nao e um erro e nao e descartado aqui; seu
efeito sobre a metrica ja foi medido em ``scripts/t04_mascaras_vazias.py``.

Uma armadilha do SSIS, nao documentada
--------------------------------------
Ausencia de deteccao chega por **duas** rotas distintas, e a segunda quebra o
codigo oficial. ``adet/modeling/ssis/condinst.py`` inicializa
``final_results = None`` em ``postprocess`` e so o atribui dentro do ramo
``if results.has("pred_global_masks")``; quando nada sobrevive, o dicionario
devolvido traz ``{"instances": None}`` em vez de um ``Instances`` vazio.

O ``demo.py`` oficial faz ``predictions["instances"].to(cpu)`` sem verificar, e
portanto lanca ``AttributeError`` nessas imagens. Isso nao aparece na
documentacao porque as amostras distribuidas pelos autores sempre tem alguma
deteccao; sobre corpus real, ocorreu na primeira centena de imagens. Aqui as
duas rotas convergem para o mesmo par preto, e o CSV registra em qual delas
cada imagem caiu (coluna ``deteccao_nula``).

Retomada
--------
Um par ja presente em disco e pulado. A execucao pode ser interrompida e
relancada sem perda, o que importa porque a extracao sobre o manifesto padrao
completo leva cerca de tres horas.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np

# Classes do SSISv2, na ordem em que o modelo as indexa. Confirmado via
# MetadataCatalog: thing_classes == ['Object', 'Shadow'].
CLASSE_OBJETO = 0
CLASSE_SOMBRA = 1

# Rotulo -> nome do diretorio, como o classificador de T04 espera encontrar:
# .../<divisao>/<classe>/<arquivo>, com a classe lida do diretorio pai
# (ver replicar_t04_object_shadow.py, CLASSE_PARA_INDICE).
ROTULO_PARA_CLASSE = {0: "real", 1: "gen"}

LADO = 256


def caminho_windows_para_wsl(bruto: str) -> Path:
    """Converte ``C:\\Users\\...`` em ``/mnt/c/Users/...``.

    O manifesto e gerado no Windows; este script le do WSL. Caminhos que ja
    sejam POSIX passam intactos.
    """
    texto = bruto.strip().replace("\\", "/")
    if len(texto) > 1 and texto[1] == ":":
        return Path(f"/mnt/{texto[0].lower()}{texto[2:]}")
    return Path(texto)


def agregar_por_classe(mascaras: np.ndarray, classes: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Reduz ``N`` mascaras de instancia a dois mapas binarios.

    O agregador que faltava. Recebe ``pred_masks`` (N, H, W) e ``pred_classes``
    (N,) e devolve ``(objeto, sombra)``, cada um (H, W) uint8 em {0, 255}.

    A reducao e a uniao logica das instancias de cada classe -- justificada pela
    inspecao das mascaras oficiais descrita no cabecalho deste modulo: elas nao
    reservam nivel de cinza para separar instancias, de modo que qualquer
    codificacao mais rica divergiria do que o classificador viu no treino.

    ``pred_masks`` chega como float32 neste modelo, e nao como bool; o limiar de
    0,5 cobre os dois casos.
    """
    if mascaras.ndim != 3:
        raise ValueError(f"esperado (N, H, W) em pred_masks, obtido {mascaras.shape}")
    if len(mascaras) != len(classes):
        raise ValueError(
            f"pred_masks tem {len(mascaras)} instancias e pred_classes tem {len(classes)}"
        )

    altura, largura = mascaras.shape[1:]
    binarias = mascaras > 0.5 if mascaras.dtype != bool else mascaras
    classes = np.asarray(classes)

    def unir(indice_classe: int) -> np.ndarray:
        selecao = binarias[classes == indice_classe]
        if len(selecao) == 0:
            # Nenhuma instancia desta classe: mapa preto. Caso legitimo e
            # frequente em imagens sem objeto com sombra projetada.
            return np.zeros((altura, largura), dtype=np.uint8)
        return (selecao.any(axis=0).astype(np.uint8)) * 255

    return unir(CLASSE_OBJETO), unir(CLASSE_SOMBRA)


def gravar_mapa(mapa: np.ndarray, destino: Path, formato: str, qualidade: int) -> None:
    """Grava um mapa binario no formato dos autores: 256x256, modo ``L``."""
    from PIL import Image

    imagem = Image.fromarray(mapa, mode="L")
    if imagem.size != (LADO, LADO):
        # NEAREST preserva a binariedade; qualquer interpolacao suave
        # introduziria os niveis intermediarios que as mascaras oficiais nao tem.
        imagem = imagem.resize((LADO, LADO), Image.NEAREST)

    destino.parent.mkdir(parents=True, exist_ok=True)
    if formato == "jpg":
        imagem.save(destino, "JPEG", quality=qualidade)
    else:
        imagem.save(destino, "PNG", optimize=True)


def ler_entradas(args) -> list[dict]:
    """Monta a lista de trabalho a partir do manifesto, ja filtrada."""
    entradas: list[dict] = []
    vistos: dict[str, str] = {}

    with open(args.manifesto, newline="", encoding="utf-8") as arquivo:
        for linha in csv.DictReader(arquivo):
            if args.split and linha.get("split") != args.split:
                continue

            rotulo = int(linha["label"])
            classe = ROTULO_PARA_CLASSE[rotulo]
            origem = caminho_windows_para_wsl(linha["path"])
            divisao = linha.get("split") or "all"
            nome = f"{origem.stem}.{args.formato}"

            # Os nomes ja vem prefixados pelo gerador (coco_000000,
            # latent_diffusion_000000); ainda assim, uma colisao silenciosa
            # sobrescreveria um mapa por outro e falsearia a avaliacao.
            chave = f"{divisao}/{classe}/{nome}"
            if chave in vistos:
                raise SystemExit(
                    f"ERRO: colisao de nome em {chave}\n"
                    f"  {vistos[chave]}\n  {origem}"
                )
            vistos[chave] = str(origem)

            entradas.append({
                "origem": origem,
                "divisao": divisao,
                "classe": classe,
                "nome": nome,
                "gerador": linha.get("generator", ""),
            })

    if args.limite:
        entradas = entradas[: args.limite]
    return entradas


def construir_preditor(args):
    """Instancia o SSISv2 pelo caminho oficial, sem a camada de visualizacao.

    ``VisualizationDemo.run_on_image`` desenha as instancias com matplotlib
    antes de devolve-las, custo que aqui e desperdicio: so as mascaras
    interessam. ``DefaultPredictor`` e exatamente o que aquela classe usa por
    dentro, de modo que a inferencia permanece a oficial.
    """
    sys.path.insert(0, str(args.ssis))
    sys.path.insert(0, str(args.ssis / "demo"))

    from adet.config import get_cfg
    from detectron2.engine.defaults import DefaultPredictor

    cfg = get_cfg()
    cfg.merge_from_file(str(args.config))
    cfg.merge_from_list(["MODEL.WEIGHTS", str(args.pesos)])
    # Mesmos limiares que demo.py aplica (demo.py:26-30), para que as
    # deteccoes coincidam com as da execucao ja verificada.
    cfg.MODEL.RETINANET.SCORE_THRESH_TEST = args.confianca
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = args.confianca
    cfg.MODEL.FCOS.INFERENCE_TH_TEST = args.confianca
    cfg.MODEL.MEInst.INFERENCE_TH_TEST = args.confianca
    cfg.MODEL.PANOPTIC_FPN.COMBINE.INSTANCES_CONFIDENCE_THRESH = args.confianca
    cfg.freeze()

    return DefaultPredictor(cfg)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Extrai mapas objeto-sombra (T04) com o SSISv2, no WSL2")
    parser.add_argument("--manifesto", type=Path, required=True,
                        help="CSV com colunas path, label, split")
    parser.add_argument("--split", default=None,
                        help="restringe a uma divisao (test, train, val)")
    parser.add_argument("--saida", type=Path, required=True,
                        help="diretorio raiz; recebe <conjunto>_shadow e <conjunto>_object")
    parser.add_argument("--conjunto", default="tcc3",
                        help="nome do conjunto nos diretorios de saida")
    parser.add_argument("--ssis", type=Path, default=Path("/root/SSIS"))
    parser.add_argument("--config", type=Path,
                        default=Path("/root/SSIS/configs/SSIS/MS_R_101_BiFPN_SSISv2_demo.yaml"))
    parser.add_argument("--pesos", type=Path,
                        default=Path("/root/SSIS/tools/output/"
                                     "SSISv2_MS_R_101_bifpn_with_offset_class_maskiouv2_da_bl/"
                                     "model_ssisv2_final.pth"))
    parser.add_argument("--confianca", type=float, default=0.1,
                        help="limiar de deteccao; 0.1 e o padrao de demo.py")
    parser.add_argument("--formato", choices=["jpg", "png"], default="jpg")
    parser.add_argument("--qualidade", type=int, default=95)
    parser.add_argument("--limite", type=int, default=0,
                        help="processa apenas as N primeiras; para testes")
    parser.add_argument("--refazer", action="store_true",
                        help="reprocessa pares ja presentes em disco")
    args = parser.parse_args()

    for caminho in (args.manifesto, args.config, args.pesos):
        if not caminho.exists():
            print(f"ERRO: nao encontrado: {caminho}")
            return 1

    entradas = ler_entradas(args)
    if not entradas:
        print("ERRO: nenhuma entrada apos o filtro de split")
        return 1

    raiz_sombra = args.saida / f"{args.conjunto}_shadow"
    raiz_objeto = args.saida / f"{args.conjunto}_object"

    print("=" * 70)
    print("T04 - EXTRACAO DE MAPAS OBJETO-SOMBRA (SSISv2)")
    print("=" * 70)
    print(f"Manifesto ....... {args.manifesto}")
    print(f"Split ........... {args.split or 'todos'}")
    print(f"Imagens ......... {len(entradas)}")
    print(f"Saida sombra .... {raiz_sombra}")
    print(f"Saida objeto .... {raiz_objeto}")
    print(f"Confianca ....... {args.confianca}")
    print(f"Formato ......... {args.formato}")
    print("=" * 70, flush=True)

    import cv2
    import torch

    preditor = construir_preditor(args)
    print("SSISv2 carregado.\n", flush=True)

    registro: list[dict] = []
    ilegiveis: list[dict] = []
    pulados = falhas = sem_deteccao = 0
    processadas = 0
    inicio = time.perf_counter()

    for indice, item in enumerate(entradas, start=1):
        rel = Path(item["divisao"]) / item["classe"] / item["nome"]
        destino_sombra = raiz_sombra / rel
        destino_objeto = raiz_objeto / rel

        if not args.refazer and destino_sombra.exists() and destino_objeto.exists():
            pulados += 1
            continue

        imagem = cv2.imread(str(item["origem"]))
        if imagem is None:
            print(f"[falha] ilegivel: {item['origem']}", flush=True)
            falhas += 1
            ilegiveis.append({"arquivo": str(rel), "erro": "cv2.imread devolveu None"})
            continue

        try:
            with torch.no_grad():
                # O modelo devolve (processed_results, processed_associations);
                # DefaultPredictor ja desempacota o primeiro nivel, e o [0]
                # restante seleciona a unica imagem do lote -- mesmo
                # encadeamento que demo/predictor.py:60 usa.
                instancias = preditor(imagem)[0]["instances"]
        except Exception as erro:                       # noqa: BLE001
            # Ha um defeito no pareamento objeto-sombra do SSIS que derruba a
            # inferencia em imagens isoladas:
            #
            #   ssis/condinst.py:321  record.pop(record[ind])
            #   ssis/condinst.py:322  record.pop(ind)      -> KeyError
            #
            # ``record`` guarda o pareamento nos dois sentidos; quando as duas
            # entradas coincidem, a primeira remocao ja apaga a chave que a
            # segunda tenta remover. Ocorreu uma vez em 42.000 imagens.
            #
            # Capturar por imagem e o que evita perder a rodada inteira por
            # causa de uma -- foi exatamente o que aconteceu na primeira
            # tentativa sobre o split de treino, a 15.358 imagens do fim.
            # A imagem e registrada e pulada, sem mapas; o avaliador so
            # considera pares completos.
            print(f"[falha] {type(erro).__name__} em {item['origem'].name}: {erro}",
                  flush=True)
            falhas += 1
            ilegiveis.append({"arquivo": str(rel), "erro": f"{type(erro).__name__}: {erro}"})
            continue

        altura, largura = imagem.shape[:2]
        if instancias is None:
            # Nao e falha. ``ssis/condinst.py:postprocess`` inicializa
            # ``final_results = None`` e so o atribui dentro do ramo
            # ``if results.has("pred_global_masks")``; quando nenhuma deteccao
            # sobrevive, o None chega ate aqui. E o mesmo caso semantico de uma
            # lista vazia de instancias, e produz o mesmo par de mapas pretos.
            #
            # O demo.py oficial nao trata este caso e quebra nestas imagens
            # (``predictions["instances"].to(...)``), o que so nao aparece na
            # documentacao porque as amostras que os autores distribuem sempre
            # tem alguma deteccao.
            mascaras = np.zeros((0, altura, largura), dtype=np.float32)
            classes = np.zeros((0,), dtype=np.int64)
            sem_deteccao += 1
        else:
            instancias = instancias.to("cpu")
            mascaras = instancias.pred_masks.numpy()
            classes = instancias.pred_classes.numpy()

        objeto, sombra = agregar_por_classe(mascaras, classes)

        gravar_mapa(sombra, destino_sombra, args.formato, args.qualidade)
        gravar_mapa(objeto, destino_objeto, args.formato, args.qualidade)

        registro.append({
            "arquivo": str(rel),
            "gerador": item["gerador"],
            "classe": item["classe"],
            "n_instancias": int(len(classes)),
            "n_objeto": int((classes == CLASSE_OBJETO).sum()),
            "n_sombra": int((classes == CLASSE_SOMBRA).sum()),
            "area_objeto": float((objeto > 0).mean()),
            "area_sombra": float((sombra > 0).mean()),
            # Distingue as duas rotas que levam a um par preto: nenhuma
            # instancia detectada, ou postprocess devolvendo None.
            "deteccao_nula": int(instancias is None),
        })
        processadas += 1

        if processadas % 250 == 0:
            decorrido = time.perf_counter() - inicio
            taxa = decorrido / processadas
            restantes = len(entradas) - indice
            print(f"  {indice}/{len(entradas)}  {taxa:.3f} s/img  "
                  f"restam ~{restantes * taxa / 60:.0f} min", flush=True)

    decorrido = time.perf_counter() - inicio
    print("\n" + "=" * 70)
    print(f"Processadas ..... {processadas}")
    print(f"Ja existentes ... {pulados}")
    print(f"Falhas .......... {falhas}")
    if processadas:
        print(f"Tempo ........... {decorrido / 60:.1f} min "
              f"({decorrido / processadas:.3f} s/img)")

    # Um arquivo por execucao, com carimbo de tempo. Anexar a um CSV anterior
    # seria enganoso: a retomada pula os pares ja gravados, entao o registro
    # cobre apenas as imagens processadas nesta execucao, nunca o corpus
    # inteiro. Os mapas em disco e que sao o artefato de referencia.
    sufixo = f"_{args.split}" if args.split else ""
    carimbo = time.strftime("%Y%m%d_%H%M%S")

    if registro:
        vazios = sum(1 for r in registro if r["n_instancias"] == 0)
        print(f"Sem instancia ... {vazios} ({vazios / len(registro):.1%})")
        print(f"  dos quais por deteccao nula: {sem_deteccao}")

        destino_csv = raiz_sombra.parent / f"{args.conjunto}_extracao{sufixo}_{carimbo}.csv"
        with open(destino_csv, "w", newline="", encoding="utf-8") as arquivo:
            escritor = csv.DictWriter(arquivo, fieldnames=list(registro[0].keys()))
            escritor.writeheader()
            escritor.writerows(registro)

        resumo = {
            "manifesto": str(args.manifesto),
            "split": args.split,
            "n_processadas": processadas,
            "n_puladas": pulados,
            "n_falhas": falhas,
            "n_sem_instancia": vazios,
            "n_deteccao_nula": sem_deteccao,
            "confianca": args.confianca,
            "formato": args.formato,
            "segundos_por_imagem": decorrido / processadas if processadas else None,
            "pesos": str(args.pesos),
        }
        destino_json = raiz_sombra.parent / f"{args.conjunto}_extracao{sufixo}_{carimbo}.json"
        destino_json.write_text(json.dumps(resumo, indent=2), encoding="utf-8")
        print(f"\nRegistro em {destino_csv}")
        print(f"Resumo em   {destino_json}")

    if ilegiveis:
        # As imagens excluidas precisam ficar auditaveis: elas nao entram na
        # avaliacao de T04, e a diferenca de n em relacao as demais tecnicas tem
        # de ser explicavel.
        destino_falhas = raiz_sombra.parent / f"{args.conjunto}_falhas{sufixo}_{carimbo}.csv"
        with open(destino_falhas, "w", newline="", encoding="utf-8") as arquivo:
            escritor = csv.DictWriter(arquivo, fieldnames=["arquivo", "erro"])
            escritor.writeheader()
            escritor.writerows(ilegiveis)
        print(f"Falhas em   {destino_falhas}")

    print("=" * 70)
    return 0 if falhas == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
