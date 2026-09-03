"""Servico residente de T04, dentro do WSL2.

Por que existe
--------------
Os tres extratores de T04 vivem no WSL2 e, ate aqui, so eram chamados em lote.
Uma chamada por imagem pagava a carga dos modelos toda vez, o que motivou a
decisao de 17/08/2026 de manter T04 fora da interface: a estimativa da epoca era
1,5 GB de modelos e 60 a 90 s de carga, acima do teto de 30 s do RNF01.

Medido em 30/08/2026, com os modelos de fato carregados:

    objeto-sombra (SSISv2)        3,4 s    +429 MiB
    campos de perspectiva (PF)    4,2 s    +414 MiB
    segmentos de reta (DeepLSD)   0,2 s     +40 MiB
    ------------------------------------------------
    total                        ~10 s     883 MiB

Ou seja, a estimativa errava o tempo por quase uma ordem de grandeza. Mantendo
os modelos residentes, o custo por imagem cai para a inferencia (~1 s somando as
tres), e o RNF01 vai de 17,871 s para cerca de 18,9 s -- com folga.

Paridade com a extracao em lote
-------------------------------
Este servico **nao reimplementa** a pontuacao: importa as funcoes dos tres
extratores e as chama na mesma ordem. Onde reimplementar era inevitavel, o
codigo esta anotado com o arquivo e a linha que ele reproduz.

O ponto que mais importa e o objeto-sombra. Na extracao em lote os mapas sao
gravados em **JPEG qualidade 95** e relidos do disco pelo classificador
(``extrair_object_shadow_ssis.py:gravar_mapa`` e depois
``avaliar_t04_corpus.py``). JPEG e com perda: ele introduz niveis de cinza
intermediarios em mapas que sairam binarios do SSISv2. Os escores publicados na
dissertacao foram medidos **depois** desse ida-e-volta.

Classificar as mascaras direto da memoria produziria numeros ligeiramente
diferentes dos do Capitulo 4 -- uma divergencia silenciosa entre a tela e o
texto. Por isso ``_escore_objeto_sombra`` reproduz o round-trip JPEG em buffer,
sem tocar o disco.

Por que sao DOIS processos
--------------------------
A extracao em lote nao usou o mesmo dispositivo para as tres representacoes --
o que so aparece nos logs, nao no codigo:

    objeto-sombra   GPU    (t04_extracao.bat)
    campos          CPU    (t04_campos.log: "Dispositivo ..... cpu")
    retas           CPU    (t04_linhas.bat, com CUDA_VISIBLE_DEVICES="")

E CPU aqui nao e opcional. O PointNet dos autores (``lines_model.py:39-41``)
move a matriz identidade para CUDA sempre que ``torch.cuda.is_available()`` for
verdadeiro, e nao quando o modelo esta na GPU; pedir CPU numa maquina com placa
levanta ``Expected all tensors to be on the same device``. O contorno registrado
em ``extrair_line_segments.py`` e esconder a placa com ``CUDA_VISIBLE_DEVICES=""``
-- que vale para o **processo inteiro**.

Logo, um unico processo nao consegue ao mesmo tempo usar a GPU para o
objeto-sombra e esconder a GPU para as retas. Servir as tres de um processo so
exigiria mudar o dispositivo de alguma delas, e a medicao de paridade mostrou
que trocar CPU por GPU nos campos altera o escore em ate 2,2e-2 -- o bastante
para a tela divergir do Capitulo 4.

Dai a divisao, com ``--representacoes``:

    porta 8404   object_shadow                      GPU
    porta 8405   perspective_fields,line_segment    CPU, CUDA_VISIBLE_DEVICES=""

Quem agrega as tres em um escore de T04 e o lado Windows.

Paridade medida, e o residuo que sobra
--------------------------------------
``automacao/verificar_paridade_servico_t04.py`` compara servico x lote sobre o
split de teste. Resultado em 30/08/2026, 15 imagens:

    campos de perspectiva   9,0e-17   exato (precisao de ponto flutuante)
    segmentos de reta       9,7e-17   exato
    objeto-sombra           2,3e-04   residuo caracterizado, abaixo

O residuo do objeto-sombra **nao** e erro de reproducao. Classificando as
mascaras que o lote gravou em disco:

    batch_size=1     0,458235770   <- igual ao servico (dif 4,6e-10)
    batch_size=128   0,458009332   <- igual a dissertacao (dif 3,3e-07)

Ou seja, a geracao de mascaras do servico e identica a do lote, e a diferenca
inteira vem da convolucao cuDNN escolher algoritmos distintos conforme o
tamanho do lote. ``avaliar_t04_corpus.py`` classifica em lotes de 128; um
servico que pontua uma imagem por vez usa lote 1, e nao ha como evitar.

Magnitude: 2,3e-04 sobre uma probabilidade. Nao move nenhum numero exibido na
tela e nao afeta a AUC de 0,5384. Fica registrado porque a alternativa seria
alguem reencontrar a diferenca depois e supor defeito.

Uma ressalva que permanece
--------------------------
O amostrador de retas (``extrair_line_segments.amostrar``) consome um
``np.random.Generator``. Em lote ha um unico gerador para a rodada inteira; aqui
cada imagem recebe um gerador novo com a mesma semente. Isso **nao** muda o
escore enquanto a imagem tiver menos de 250 retas -- so o ramo de duplicacao
roda, e o max-pooling do PointNet e insensivel a duplicatas, como o proprio
docstring de ``amostrar`` registra. Acima de 250 retas o escore passa a depender
da amostragem, e ai lote e servico podem divergir. Nas medias deste corpus (49 a
80 retas) o caso nao ocorre; em uma imagem arbitraria enviada pela tela, pode.

Uso (dentro do WSL2)::

    /root/geo/bin/python servico_t04.py --porta 8404 --dispositivo cuda

Verificacao de saude::

    curl -s http://127.0.0.1:8404/saude

Pontuacao de uma imagem::

    curl -s -X POST http://127.0.0.1:8404/escore \\
         -H 'Content-Type: application/json' \\
         -d '{"caminho": "/mnt/c/Users/ferre/.../imagem.png"}'
"""

from __future__ import annotations

import argparse
import io
import json
import sys
import threading
import time
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import cv2
import numpy as np
import torch
import torchvision
from PIL import Image

AQUI = Path(__file__).resolve().parent
sys.path.insert(0, str(AQUI))

# As funcoes de pontuacao vem dos extratores, nao de copias. Qualquer correcao
# neles passa a valer aqui sem intervencao.
from extrair_line_segments import (                       # noqa: E402
    CONF_DETECCAO, N_RETAS, amostrar,
    carregar_classificador as _carregar_clf_retas,
    carregar_detector as _carregar_detector_retas,
)
from extrair_object_shadow_ssis import (                  # noqa: E402
    LADO, agregar_por_classe,
)
from extrair_perspective_fields import (                  # noqa: E402
    montar_entrada, carregar_classificador as _carregar_clf_campos,
)

RAIZ = Path("/mnt/c/Users/ferre/projects/tcc3-deteccao-imagens-ia")
PESOS_T04 = RAIZ / "pesos" / "projective_geometry"
# Mesma arquitetura que extrair_line_segments.py:180 resolve: o PointNet oficial
# vem do repositorio clonado em externo/, carregado por caminho.
ARQUITETURA_RETAS = RAIZ / "externo" / "projective-geometry" / "line_segment" / "lines_model.py"
QUALIDADE_JPEG = 95          # extrair_object_shadow_ssis.py:249, default --qualidade
CONFIANCA_SSIS = 0.1         # extrair_object_shadow_ssis.py:246, default --confianca
SEMENTE = 42                 # extrair_line_segments.py:175, default --semente


# ---------------------------------------------------------------------------
# Carga dos modelos
# ---------------------------------------------------------------------------

def _carregar_clf_objeto_sombra(pesos: Path, dispositivo):
    """ResNet-50 com primeira convolucao de 2 bandas.

    Reproduz ``automacao/replicar_t04_object_shadow.py:carregar_modelo``, que por
    sua vez segue ``test.py:112-114`` dos autores. Nao e importado de la porque
    aquele modulo puxa ``codigo.configuracao``, que pertence ao lado Windows do projeto.
    """
    import torch.nn as nn

    modelo = torchvision.models.resnet50(weights=None)
    modelo.conv1 = nn.Conv2d(2, 64, kernel_size=(7, 7), stride=(2, 2),
                             padding=(3, 3), bias=False)
    modelo.fc = nn.Linear(in_features=2048, out_features=2, bias=True)
    modelo.load_state_dict(torch.load(pesos, map_location=dispositivo))
    modelo.to(dispositivo)
    modelo.eval()
    return modelo


def _carregar_ssis(dispositivo: str):
    """SSISv2 via ``DefaultPredictor``, como no extrator em lote."""
    sys.path.insert(0, "/root/SSIS")
    sys.path.insert(0, "/root/SSIS/demo")
    from adet.config import get_cfg
    from detectron2.engine.defaults import DefaultPredictor

    cfg = get_cfg()
    cfg.merge_from_file("/root/SSIS/configs/SSIS/MS_R_101_BiFPN_SSISv2_demo.yaml")
    cfg.merge_from_list([
        "MODEL.WEIGHTS",
        "/root/SSIS/tools/output/"
        "SSISv2_MS_R_101_bifpn_with_offset_class_maskiouv2_da_bl/model_ssisv2_final.pth",
    ])
    # Mesmos limiares de demo.py:26-30, para que as deteccoes coincidam com as
    # da extracao ja verificada.
    for chave in ("MODEL.RETINANET.SCORE_THRESH_TEST",
                  "MODEL.ROI_HEADS.SCORE_THRESH_TEST",
                  "MODEL.FCOS.INFERENCE_TH_TEST",
                  "MODEL.MEInst.INFERENCE_TH_TEST"):
        cfg.merge_from_list([chave, CONFIANCA_SSIS])
    cfg.MODEL.PANOPTIC_FPN.COMBINE.INSTANCES_CONFIDENCE_THRESH = CONFIANCA_SSIS
    cfg.MODEL.DEVICE = dispositivo
    cfg.freeze()
    return DefaultPredictor(cfg)


class Modelos:
    """Os tres detectores e os tres classificadores, residentes."""

    def __init__(self, variante: str, dispositivo: str,
                 representacoes: set[str]) -> None:
        self.dispositivo = dispositivo
        self.dispositivo_retas = dispositivo
        self.variante = variante
        self.representacoes = representacoes
        self.ssis = self.clf_objeto_sombra = None
        self.pf = self.clf_campos = None
        self.deeplsd = self.clf_retas = None
        # Uma imagem por vez: os modelos nao sao reentrantes e a placa e uma so.
        self.trava = threading.Lock()

        if "object_shadow" in representacoes:
            t0 = time.time()
            self.ssis = _carregar_ssis(dispositivo)
            self.clf_objeto_sombra = _carregar_clf_objeto_sombra(
                PESOS_T04 / "object_shadow" / f"ShadowObject_{variante}.pth",
                torch.device(dispositivo))
            print(f"[carga] objeto-sombra ....... {time.time() - t0:5.1f}s", flush=True)

        if "perspective_fields" in representacoes:
            t0 = time.time()
            from perspective2d import PerspectiveFields
            self.pf = PerspectiveFields("Paramnet-360Cities-edina-centered").eval()
            if dispositivo == "cuda":
                self.pf = self.pf.cuda()
            self.clf_campos = _carregar_clf_campos(
                PESOS_T04 / "perspective_fields" / f"Fields_{variante}.pt",
                torch.device(dispositivo))
            print(f"[carga] campos ............... {time.time() - t0:5.1f}s", flush=True)

        if "line_segment" in representacoes:
            t0 = time.time()
            self.deeplsd = _carregar_detector_retas(
                Path("/root/DeepLSD/weights/deeplsd_md.tar"), dispositivo)
            self.clf_retas = _carregar_clf_retas(
                PESOS_T04 / "line_segment" / f"Lines_{variante}.pt",
                ARQUITETURA_RETAS,
                torch.device(dispositivo))
            print(f"[carga] retas ................ {time.time() - t0:5.1f}s", flush=True)


# ---------------------------------------------------------------------------
# Pontuacao, uma representacao por funcao
# ---------------------------------------------------------------------------

def _mapa_pela_ida_e_volta_jpeg(mapa: np.ndarray) -> Image.Image:
    """Aplica a ``mapa`` exatamente o que ``gravar_mapa`` + releitura aplicam.

    Reproduz ``extrair_object_shadow_ssis.py:gravar_mapa`` (redimensiona para
    256x256 com NEAREST, grava JPEG q95) seguido da leitura em modo ``L`` que
    ``ConjuntoObjetoSombra.__getitem__`` faz. O buffer substitui o disco; o
    resultado em pixels e o mesmo, incluindo a perda do JPEG.
    """
    imagem = Image.fromarray(mapa, mode="L")
    if imagem.size != (LADO, LADO):
        imagem = imagem.resize((LADO, LADO), Image.NEAREST)
    buffer = io.BytesIO()
    imagem.save(buffer, "JPEG", quality=QUALIDADE_JPEG)
    buffer.seek(0)
    return Image.open(buffer).convert("L")


def escore_objeto_sombra(m: Modelos, imagem_bgr: np.ndarray) -> tuple[float | None, bool]:
    """Devolve (escore, par_vazio). ``None`` quando a inferencia falha."""
    with torch.no_grad():
        instancias = m.ssis(imagem_bgr)[0]["instances"]

    altura, largura = imagem_bgr.shape[:2]
    if instancias is None:
        # postprocess devolve None quando nenhuma deteccao sobrevive; e o mesmo
        # caso semantico de lista vazia (extrair_object_shadow_ssis.py:340).
        mascaras = np.zeros((0, altura, largura), dtype=np.float32)
        classes = np.zeros((0,), dtype=np.int64)
    else:
        instancias = instancias.to("cpu")
        mascaras = instancias.pred_masks.numpy()
        classes = instancias.pred_classes.numpy()

    objeto, sombra = agregar_por_classe(mascaras, classes)
    par_vazio = bool(objeto.max() == 0 and sombra.max() == 0)

    para_tensor = torchvision.transforms.ToTensor()
    banda_sombra = para_tensor(_mapa_pela_ida_e_volta_jpeg(sombra))
    banda_objeto = para_tensor(_mapa_pela_ida_e_volta_jpeg(objeto))
    entrada = torch.cat([banda_sombra, banda_objeto], dim=0).unsqueeze(0)

    with torch.no_grad():
        saida = m.clf_objeto_sombra(entrada.to(m.dispositivo))
        # Coluna 1 = classe "gen" (CLASSE_PARA_INDICE em replicar_t04_object_shadow).
        return float(torch.softmax(saida, dim=1)[0, 1]), par_vazio


def escore_campos(m: Modelos, imagem_bgr: np.ndarray) -> float:
    with torch.no_grad():
        campo = m.pf.inference(img_bgr=imagem_bgr)
        entrada = montar_entrada(campo).unsqueeze(0).to(m.dispositivo)
        saida = m.clf_campos(entrada)
        return float(torch.softmax(saida, dim=1)[0, 1])


def escore_retas(m: Modelos, imagem_bgr: np.ndarray) -> tuple[float | None, int]:
    """Devolve (escore, n_retas). ``None`` quando nenhuma reta e detectada."""
    cinza = cv2.cvtColor(imagem_bgr, cv2.COLOR_BGR2GRAY)
    entrada = {"image": torch.tensor(cinza, dtype=torch.float,
                                     device=m.dispositivo_retas)[None, None] / 255.0}
    with torch.no_grad():
        retas = np.asarray(m.deeplsd(entrada)["lines"][0], dtype=np.float32)

    if retas.size == 0:
        return None, 0
    retas = retas.reshape(retas.shape[0], -1)              # (N, 2, 2) -> (N, 4)

    gerador = np.random.default_rng(SEMENTE)
    with torch.no_grad():
        lote = torch.from_numpy(amostrar(retas, gerador)[None]).to(m.dispositivo_retas)
        saida = m.clf_retas(lote)
        return float(torch.softmax(saida, dim=1)[0, 1]), int(retas.shape[0])


def pontuar(m: Modelos, caminho: Path) -> dict:
    """As tres representacoes e a media, para uma imagem."""
    imagem = cv2.imread(str(caminho))
    if imagem is None:
        raise ValueError(f"cv2.imread devolveu None para {caminho}")

    resultado: dict = {"caminho": str(caminho)}
    inicio = time.perf_counter()

    # Cada representacao falha por conta propria: uma que quebre nao derruba as
    # outras, do mesmo modo que a extracao em lote registra e segue (RN07).
    for nome, funcao in (("object_shadow", "os"), ("perspective_fields", "pf"),
                         ("line_segment", "ls")):
        if nome not in m.representacoes:
            continue                                       # nao servida por esta instancia
        t0 = time.perf_counter()
        try:
            if funcao == "os":
                valor, vazio = escore_objeto_sombra(m, imagem)
                resultado["par_vazio"] = vazio
            elif funcao == "pf":
                valor = escore_campos(m, imagem)
            else:
                valor, n = escore_retas(m, imagem)
                resultado["n_retas"] = n
            resultado[nome] = valor
        except Exception as erro:                          # noqa: BLE001
            resultado[nome] = None
            resultado.setdefault("falhas", {})[nome] = f"{type(erro).__name__}: {erro}"
            print(f"[falha] {nome}: {erro}", flush=True)
        resultado[f"ms_{nome}"] = round((time.perf_counter() - t0) * 1000, 1)

    validos = [resultado[k] for k in ("object_shadow", "perspective_fields", "line_segment")
               if resultado.get(k) is not None]
    # Media apenas do que ESTA instancia serve. Com o servico dividido em dois
    # processos (ver --representacoes), a media das tres cabe a quem agrega, no
    # lado Windows -- e o mesmo nanmean de T04ProjectiveGeometry.predict_proba.
    resultado["parcial"] = float(np.mean(validos)) if validos else None
    resultado["ms_total"] = round((time.perf_counter() - inicio) * 1000, 1)
    return resultado


# ---------------------------------------------------------------------------
# Servidor
# ---------------------------------------------------------------------------

class Manipulador(BaseHTTPRequestHandler):
    modelos: Modelos = None                                # preenchido em servir()

    def _responder(self, codigo: int, corpo: dict) -> None:
        dados = json.dumps(corpo).encode("utf-8")
        self.send_response(codigo)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(dados)))
        self.end_headers()
        self.wfile.write(dados)

    def do_GET(self) -> None:                              # noqa: N802
        if self.path.rstrip("/") in ("/saude", ""):
            self._responder(200, {
                "ok": True,
                "variante": self.modelos.variante,
                "dispositivo": self.modelos.dispositivo,
                "cuda_visivel": torch.cuda.is_available(),
                "representacoes": sorted(self.modelos.representacoes),
            })
        else:
            self._responder(404, {"erro": "rota desconhecida"})

    def do_POST(self) -> None:                             # noqa: N802
        if self.path.rstrip("/") != "/escore":
            self._responder(404, {"erro": "rota desconhecida"})
            return
        try:
            tamanho = int(self.headers.get("Content-Length", 0))
            pedido = json.loads(self.rfile.read(tamanho) or b"{}")
            caminho = Path(pedido["caminho"])
            if not caminho.exists():
                self._responder(400, {"erro": f"arquivo inexistente: {caminho}"})
                return
            # Serializa: os modelos nao sao reentrantes e a GPU e compartilhada.
            with self.modelos.trava:
                self._responder(200, pontuar(self.modelos, caminho))
        except Exception as erro:                          # noqa: BLE001
            traceback.print_exc()
            self._responder(500, {"erro": f"{type(erro).__name__}: {erro}"})

    def log_message(self, formato, *args) -> None:         # noqa: A002
        pass                                               # silencia o log por requisicao


def servir(porta: int, modelos: Modelos) -> None:
    Manipulador.modelos = modelos
    # 0.0.0.0 e o que torna o servico alcancavel do Windows: o WSL2 encaminha
    # localhost para dentro da VM, mas so para portas ligadas em todas as
    # interfaces.
    servidor = ThreadingHTTPServer(("0.0.0.0", porta), Manipulador)
    print(f"\n[pronto] servico de T04 em http://0.0.0.0:{porta}", flush=True)
    print("[pronto] rotas: GET /saude   POST /escore", flush=True)
    servidor.serve_forever()


def main() -> int:
    parser = argparse.ArgumentParser(description="Servico residente de T04 (WSL2)")
    parser.add_argument("--porta", type=int, default=8404)
    parser.add_argument("--variante", default="combined",
                        choices=["combined", "indoor", "outdoor"])
    parser.add_argument("--dispositivo", default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--representacoes", default="todas",
                        help="lista separada por virgula, ou 'todas'. "
                             "Ver a nota sobre a divisao em dois processos.")
    args = parser.parse_args()

    todas = {"object_shadow", "perspective_fields", "line_segment"}
    if args.representacoes == "todas":
        representacoes = todas
    else:
        representacoes = {r.strip() for r in args.representacoes.split(",") if r.strip()}
        desconhecidas = representacoes - todas
        if desconhecidas:
            print(f"ERRO: representacao desconhecida: {sorted(desconhecidas)}")
            return 1

    print(f"variante ......... {args.variante}")
    print(f"dispositivo ...... {args.dispositivo}")
    print(f"representacoes ... {', '.join(sorted(representacoes))}")
    print(f"cuda visivel ..... {torch.cuda.is_available()}\n")

    modelos = Modelos(args.variante, args.dispositivo, representacoes)
    servir(args.porta, modelos)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
