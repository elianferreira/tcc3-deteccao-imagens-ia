"""Segundo componente de T04: campos de perspectiva, extraidos e classificados.

Fecha a segunda das tres representacoes geometricas de que T04 depende. A
primeira -- objeto-sombra -- foi resolvida em
``automacao/wsl/extrair_object_shadow_ssis.py``; esta usa o mesmo ambiente WSL2
descrito em ``documentacao/T04_AMBIENTE_WSL2.md``.

O extrator
----------
PerspectiveFields (Jin et al., 2023), o modelo que Sarkar et al. (2024) citam
como origem dos campos. Instalado por::

    /root/geo/bin/pip install git+https://github.com/jinlinyi/PerspectiveFields.git

Contraste com o objeto-sombra, que custou oito obstaculos nao documentados: aqui
bastou o pip e uma chamada de API. O checkpoint (798 MB) e baixado sozinho na
primeira execucao.

O contrato, verificado
----------------------
``fields_dataset.py:31-35`` carrega um ``.pt`` por imagem e le duas chaves. A
saida de ``PerspectiveFields.inference`` traz exatamente essas duas, com as
formas certas::

    pred_latitude_original   (H, W)      float32, em graus
    pred_gravity_original    (2, H, W)   float32

E o dataset oficial as combina em tres canais::

    cat([latitude/90.0, gravity], dim=0)

Por que classificar na mesma passagem
-------------------------------------
Gravar os campos em disco seria o espelho do que se fez com as mascaras, mas
nao cabe: sao 768 KB por imagem em float32, ou **47 GB** para as 60.000 do
manifesto padrao, contra 27 GB livres. Nem float16 resolve.

Entao o campo e consumido em memoria e so o escore vai para disco. O
classificador e o oficial (``fields_model.py``: ResNet-50 com ``fc`` de duas
saidas, entrada de tres canais), com os pesos oficiais, sem reajuste --
conformidade com a Etapa 2.

``--amostra-campos N`` grava os N primeiros campos em disco, para auditoria da
representacao sem pagar o custo de todos.

Incerteza registrada
--------------------
Sarkar et al. dizem apenas "we generate these fields from single images using a
pretrained model", sem nomear o checkpoint. Aqui usa-se
``Paramnet-360Cities-edina-centered``, o padrao recomendado pelos autores do
PerspectiveFields, que cobre cenas internas e externas. Se os autores de T04
usaram outro, os campos diferem em detalhe -- e isso limita a comparacao com o
valor publicado, do mesmo modo que a leitura de figura ja limitava.

Uso (dentro do WSL2)::

    /root/geo/bin/python extrair_perspective_fields.py \
        --manifesto .../manifesto30k_standard.csv --split test --variante combined
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np

# Rotulo -> nome de classe, como no restante da replicacao de T04.
ROTULO_PARA_CLASSE = {0: "real", 1: "gen"}

VERSAO_EXTRATOR = "Paramnet-360Cities-edina-centered"


def caminho_windows_para_wsl(bruto: str) -> Path:
    """Converte ``C:\\Users\\...`` em ``/mnt/c/Users/...``."""
    texto = bruto.strip().replace("\\", "/")
    if len(texto) > 1 and texto[1] == ":":
        return Path(f"/mnt/{texto[0].lower()}{texto[2:]}")
    return Path(texto)


def montar_entrada(campo: dict) -> "torch.Tensor":       # noqa: F821
    """Combina latitude e gravidade nos tres canais que o classificador espera.

    Reproduz ``PerspectiveMapDataset.transform_maps``
    (``fields_dataset.py:13-16``): a latitude vem em graus e e normalizada por
    90, a gravidade entra como esta, e as duas sao concatenadas no eixo de
    canais.
    """
    import torch

    latitude = campo["pred_latitude_original"]
    gravidade = campo["pred_gravity_original"]

    if latitude.ndim != 2:
        raise ValueError(f"latitude com forma {tuple(latitude.shape)}, esperado (H, W)")
    if gravidade.ndim != 3 or gravidade.shape[0] != 2:
        raise ValueError(
            f"gravidade com forma {tuple(gravidade.shape)}, esperado (2, H, W)")

    return torch.cat([(latitude / 90.0).unsqueeze(0), gravidade], dim=0)


def carregar_classificador(pesos: Path, dispositivo):
    """Reproduz ``FieldsClassifier`` de ``fields_model.py``.

    A ResNet fica **dentro** de um atributo ``self.resnet``, e nao e o modulo de
    topo. Isso aparece nas chaves do checkpoint, todas prefixadas por
    ``resnet.``; carregar direto em um ``resnet50`` falha com centenas de chaves
    faltando. A estrutura precisa ser a mesma para que os pesos oficiais entrem
    sem remapeamento -- remapear seria uma reinterpretacao do modelo dos autores,
    e nao e necessario.
    """
    import torch
    import torch.nn as nn
    from torchvision.models import resnet50

    class FieldsClassifier(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.resnet = resnet50(weights=None)
            self.resnet.fc = nn.Linear(self.resnet.fc.in_features, 2)

        def forward(self, x):
            return self.resnet(x)

    modelo = FieldsClassifier()
    estado = torch.load(pesos, map_location=dispositivo)
    modelo.load_state_dict(estado)
    modelo.to(dispositivo)
    modelo.eval()
    return modelo


def gravar_bloco(destino: Path, linhas: list[dict]) -> None:
    """Acrescenta um bloco de resultados ao CSV, criando o cabecalho se preciso.

    A gravacao e incremental para que a execucao seja retomavel. Sem isso, uma
    interrupcao no meio custa tudo que foi processado -- o que custou 1.250
    imagens quando a maquina precisou ser liberada.
    """
    if not linhas:
        return
    novo = not destino.exists()
    with open(destino, "a", newline="", encoding="utf-8") as arquivo:
        escritor = csv.DictWriter(arquivo, fieldnames=list(linhas[0].keys()))
        if novo:
            escritor.writeheader()
        escritor.writerows(linhas)


def ler_entradas(manifesto: Path, split: str | None, limite: int) -> list[dict]:
    entradas: list[dict] = []
    with open(manifesto, newline="", encoding="utf-8") as arquivo:
        for linha in csv.DictReader(arquivo):
            if split and linha.get("split") != split:
                continue
            origem = caminho_windows_para_wsl(linha["path"])
            entradas.append({
                "origem": origem,
                "nome": origem.stem,
                "divisao": linha.get("split") or "all",
                "classe": ROTULO_PARA_CLASSE[int(linha["label"])],
                "rotulo": int(linha["label"]),
                "gerador": linha.get("generator", ""),
            })
    return entradas[:limite] if limite else entradas


def main() -> int:
    parser = argparse.ArgumentParser(
        description="T04, componente de campos de perspectiva")
    parser.add_argument("--manifesto", type=Path, required=True)
    parser.add_argument("--split", default=None)
    parser.add_argument("--saida", type=Path, required=True,
                        help="diretorio para os CSV de escores")
    parser.add_argument("--conjunto", default="tcc3_30k")
    parser.add_argument("--variante", default="combined",
                        choices=["combined", "indoor", "outdoor"])
    parser.add_argument("--pesos-raiz", type=Path,
                        default=Path("/mnt/c/Users/ferre/projects/"
                                     "tcc3-deteccao-imagens-ia/weights/"
                                     "projective_geometry/perspective_fields"))
    parser.add_argument("--versao-extrator", default=VERSAO_EXTRATOR)
    parser.add_argument("--dispositivo", default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--limite", type=int, default=0)
    parser.add_argument("--amostra-campos", type=int, default=0,
                        help="grava os N primeiros campos para auditoria")
    args = parser.parse_args()

    pesos = args.pesos_raiz / f"Fields_{args.variante}.pt"
    for caminho in (args.manifesto, pesos):
        if not caminho.exists():
            print(f"ERRO: nao encontrado: {caminho}")
            return 1

    entradas = ler_entradas(args.manifesto, args.split, args.limite)
    if not entradas:
        print("ERRO: nenhuma entrada apos o filtro de split")
        return 1

    args.saida.mkdir(parents=True, exist_ok=True)
    sufixo = f"_{args.split}" if args.split else ""
    destino_csv = (args.saida /
                   f"{args.conjunto}_pf_{args.variante}{sufixo}.csv")

    # Retomada por checkpoint. O CSV e gravado em blocos, e nao so no fim: uma
    # interrupcao a 1.250 de 9.000 imagens custava as 1.250 inteiras, o que
    # aconteceu de fato ao liberar a maquina no meio de uma execucao.
    ja_feitas: set[str] = set()
    if destino_csv.exists():
        with open(destino_csv, newline="", encoding="utf-8") as arquivo:
            ja_feitas = {linha["arquivo"] for linha in csv.DictReader(arquivo)}
        if len(ja_feitas) >= len(entradas):
            print(f"[retomada] {destino_csv.name} completo ({len(ja_feitas)} imagens)")
            return 0
        print(f"[retomada] {len(ja_feitas)} imagens ja processadas; continuando\n")
        entradas = [e for e in entradas if e["nome"] not in ja_feitas]

    print("=" * 72)
    print("T04 - CAMPOS DE PERSPECTIVA (extracao + classificacao)")
    print("=" * 72)
    print(f"Manifesto ....... {args.manifesto}")
    print(f"Split ........... {args.split or 'todos'}")
    print(f"Imagens ......... {len(entradas)}")
    print(f"Extrator ........ {args.versao_extrator}")
    print(f"Classificador ... {pesos.name}")
    print(f"Dispositivo ..... {args.dispositivo}")
    print("=" * 72, flush=True)

    import cv2
    import torch
    from perspective2d import PerspectiveFields

    extrator = PerspectiveFields(args.versao_extrator).eval()
    if args.dispositivo == "cuda":
        extrator = extrator.cuda()
    classificador = carregar_classificador(pesos, torch.device(args.dispositivo))
    print("Modelos carregados.\n", flush=True)

    registro: list[dict] = []
    falhas: list[dict] = []
    inicio = time.perf_counter()

    for indice, item in enumerate(entradas, start=1):
        imagem = cv2.imread(str(item["origem"]))
        if imagem is None:
            falhas.append({"arquivo": item["nome"], "erro": "cv2.imread devolveu None"})
            continue

        try:
            with torch.no_grad():
                campo = extrator.inference(img_bgr=imagem)
                entrada = montar_entrada(campo).unsqueeze(0).to(args.dispositivo)
                saida = classificador(entrada)
                # Coluna 1 = classe "gen", como no test oficial.
                escore = float(torch.softmax(saida, dim=1)[0, 1])
        except Exception as erro:                       # noqa: BLE001
            # Mesma politica do extrator de objeto-sombra: uma imagem que falha
            # nao pode derrubar a rodada inteira.
            print(f"[falha] {type(erro).__name__} em {item['nome']}: {erro}", flush=True)
            falhas.append({"arquivo": item["nome"],
                           "erro": f"{type(erro).__name__}: {erro}"})
            continue

        if args.amostra_campos and len(registro) < args.amostra_campos:
            amostra = args.saida / "campos_amostra"
            amostra.mkdir(parents=True, exist_ok=True)
            torch.save(
                {"pred_latitude_original": campo["pred_latitude_original"].cpu(),
                 "pred_gravity_original": campo["pred_gravity_original"].cpu()},
                amostra / f"{item['nome']}.pt",
            )

        registro.append({
            "arquivo": item["nome"],
            "split": item["divisao"],
            "label": item["rotulo"],
            "gerador": item["gerador"],
            "escore_t04_pf": escore,
        })

        if len(registro) % 250 == 0:
            gravar_bloco(destino_csv, registro[-250:])
            decorrido = time.perf_counter() - inicio
            taxa = decorrido / len(registro)
            restantes = len(entradas) - indice
            print(f"  {indice}/{len(entradas)}  {taxa:.3f} s/img  "
                  f"restam ~{restantes * taxa / 60:.0f} min", flush=True)

    decorrido = time.perf_counter() - inicio
    print("\n" + "=" * 72)
    print(f"Processadas ..... {len(registro)}")
    print(f"Falhas .......... {len(falhas)}")
    if registro:
        print(f"Tempo ........... {decorrido / 60:.1f} min "
              f"({decorrido / len(registro):.3f} s/img)")

        # Grava o resto do ultimo bloco, que nao fechou o multiplo de 250.
        gravados = (len(registro) // 250) * 250
        gravar_bloco(destino_csv, registro[gravados:])
        print(f"\nEscores em {destino_csv}")

        # A AUC e calculada sobre o arquivo completo, e nao so sobre esta
        # execucao: numa retomada, ``registro`` tem apenas a parte nova.
        with open(destino_csv, newline="", encoding="utf-8") as arquivo:
            todas = list(csv.DictReader(arquivo))
        escores = np.array([float(r["escore_t04_pf"]) for r in todas])
        rotulos = np.array([int(r["label"]) for r in todas])
        print(f"Total no arquivo: {len(todas)} imagens")
        if len(np.unique(rotulos)) == 2:
            from sklearn.metrics import roc_auc_score
            print(f"AUC deste split: {roc_auc_score(rotulos, escores):.4f}")

    if falhas:
        destino_falhas = args.saida / f"{args.conjunto}_pf_falhas{sufixo}.csv"
        with open(destino_falhas, "w", newline="", encoding="utf-8") as arquivo:
            escritor = csv.DictWriter(arquivo, fieldnames=["arquivo", "erro"])
            escritor.writeheader()
            escritor.writerows(falhas)
        print(f"Falhas em  {destino_falhas}")

    resumo = {
        "manifesto": str(args.manifesto), "split": args.split,
        "extrator": args.versao_extrator, "classificador": str(pesos),
        "n_processadas": len(registro), "n_falhas": len(falhas),
        "segundos_por_imagem": decorrido / len(registro) if registro else None,
    }
    (args.saida / f"{args.conjunto}_pf_{args.variante}{sufixo}.json").write_text(
        json.dumps(resumo, indent=2), encoding="utf-8")
    print("=" * 72)
    return 0 if not falhas else 1


if __name__ == "__main__":
    raise SystemExit(main())
