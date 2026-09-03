"""Grad-CAM para T01, previsto na Etapa 5 do TCC 2.

O que este mapa mostra -- e o que ele NAO mostra
------------------------------------------------
T01 nao recebe a imagem: recebe as **matrizes de coocorrencia** dos canais R, G
e B (``rgb_cooccurrence_tensor``). Nessa representacao, a posicao ``(i, j)``
acumula quantas vezes um pixel de intensidade ``i`` aparece adjacente a um pixel
de intensidade ``j``. Os dois eixos sao niveis de intensidade, de 0 a 255 --
**nao** sao coordenadas espaciais da fotografia.

Em consequencia, o Grad-CAM de T01 responde a pergunta "quais transicoes de
intensidade pesaram na decisao", e nao "que regiao da imagem parece sintetica".
Sobrepor este mapa a fotografia seria uma leitura incorreta: nao existe
correspondencia posicional entre os dois. A regiao proxima da diagonal
principal concentra transicoes suaves (pixels vizinhos de intensidade parecida,
tipicas de areas homogeneas); regioes afastadas da diagonal correspondem a
bordas e transicoes abruptas.

Essa e uma diferenca relevante em relacao ao uso usual de Grad-CAM em
classificadores que operam sobre pixels, e precisa constar da legenda da figura
no Capitulo 4.

Camada alvo
-----------
``features[13]`` -- saida da ultima convolucao (128 canais) apos ReLU, antes do
terceiro max pooling, com resolucao 64 x 64 no espaco de coocorrencia.

Uso::

    python automacao/gradcam_t01.py --imagens <caminho> [<caminho> ...]
    python automacao/gradcam_t01.py --amostrar-por-gerador 1
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib                                    # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt                      # noqa: E402
from PIL import Image                                # noqa: E402

from codigo.configuracao import DATA_DIR, RESULTS_DIR, WEIGHTS_DIR      # noqa: E402
from codigo.tecnicas.t01_coocorrencia import (                  # noqa: E402
    CooccurrenceDataset, T01Cooccurrence, rgb_cooccurrence_tensor,
)

CAMADA_ALVO = 13     # ultima ReLU do bloco convolucional


class GradCAM:
    """Grad-CAM sobre a ultima camada convolucional de T01.

    Registra ganchos de ativacao e gradiente, propaga o logit da classe
    "sintetica" e pondera os mapas de ativacao pelos gradientes medios.
    """

    def __init__(self, modelo: torch.nn.Module, indice_camada: int = CAMADA_ALVO):
        self.modelo = modelo
        self.ativacoes: torch.Tensor | None = None
        self.gradientes: torch.Tensor | None = None

        # A rede usa ReLU com inplace=True, o que economiza memoria no
        # treinamento mas e incompativel com full_backward_hook: o PyTorch
        # recusa modificar in-place a saida capturada pelo gancho, sob pena de
        # gradientes incorretos. Desativar inplace altera apenas o consumo de
        # memoria; a saida numerica e identica.
        for modulo in modelo.modules():
            if isinstance(modulo, torch.nn.ReLU):
                modulo.inplace = False

        camada = modelo.features[indice_camada]
        camada.register_forward_hook(self._guardar_ativacao)
        # full_backward_hook entrega o gradiente em relacao a saida da camada.
        camada.register_full_backward_hook(self._guardar_gradiente)

    def _guardar_ativacao(self, _modulo, _entrada, saida):
        self.ativacoes = saida.detach()

    def _guardar_gradiente(self, _modulo, _grad_entrada, grad_saida):
        self.gradientes = grad_saida[0].detach()

    def gerar(self, entrada: torch.Tensor) -> tuple[np.ndarray, float]:
        self.modelo.zero_grad(set_to_none=True)
        logit = self.modelo(entrada)
        # Um unico neuronio de saida: o logit ja e o escore da classe sintetica.
        logit.sum().backward()

        if self.ativacoes is None or self.gradientes is None:
            raise RuntimeError("ganchos nao capturaram ativacao ou gradiente")

        # Peso de cada canal = gradiente medio espacial (Selvaraju et al., 2017).
        pesos = self.gradientes.mean(dim=(2, 3), keepdim=True)
        mapa = F.relu((pesos * self.ativacoes).sum(dim=1, keepdim=True))
        mapa = F.interpolate(mapa, size=entrada.shape[-2:],
                             mode="bilinear", align_corners=False)

        mapa = mapa[0, 0].cpu().numpy()
        if mapa.max() > mapa.min():
            mapa = (mapa - mapa.min()) / (mapa.max() - mapa.min())
        return mapa, float(torch.sigmoid(logit)[0].item())


def figura(caminho: Path, coocorrencia: np.ndarray, mapa: np.ndarray,
           probabilidade: float, destino: Path) -> None:
    figura, eixos = plt.subplots(1, 3, figsize=(15, 5))

    with Image.open(caminho) as imagem:
        eixos[0].imshow(imagem.convert("RGB"))
    eixos[0].set_title(f"Imagem\n{caminho.name}", fontsize=10)
    eixos[0].axis("off")

    # Escala logaritmica: a coocorrencia concentra massa na diagonal e sem ela
    # o restante do plano fica invisivel.
    media = coocorrencia.mean(axis=0)
    eixos[1].imshow(np.log1p(media * 1e4), cmap="viridis", origin="lower")
    eixos[1].set_title("Coocorrência média RGB\n(escala log)", fontsize=10)
    eixos[1].set_xlabel("intensidade do pixel deslocado")
    eixos[1].set_ylabel("intensidade do pixel de referência")

    eixos[2].imshow(np.log1p(media * 1e4), cmap="gray", origin="lower")
    imagem_mapa = eixos[2].imshow(mapa, cmap="jet", alpha=0.5, origin="lower")
    eixos[2].set_title(f"Grad-CAM sobre a coocorrência\nP(sintética) = "
                       f"{probabilidade * 100:.1f}%", fontsize=10)
    eixos[2].set_xlabel("intensidade do pixel deslocado")
    eixos[2].set_ylabel("intensidade do pixel de referência")
    figura.colorbar(imagem_mapa, ax=eixos[2], fraction=0.046)

    figura.suptitle(
        "Os eixos dos painéis 2 e 3 são níveis de intensidade, não posições na imagem: "
        "o mapa indica quais transições pesaram na decisão.",
        fontsize=9, y=0.02,
    )
    figura.tight_layout(rect=(0, 0.05, 1, 1))
    figura.savefig(destino, dpi=150, bbox_inches="tight")
    plt.close(figura)


def main() -> int:
    parser = argparse.ArgumentParser(description="Grad-CAM de T01 (Etapa 5)")
    parser.add_argument("--imagens", type=Path, nargs="*", default=[])
    parser.add_argument("--amostrar-por-gerador", type=int, default=0,
                        help="amostra N imagens de cada gerador do corpus")
    parser.add_argument("--corpus", type=Path, default=DATA_DIR / "corvi2024_30k_norm")
    parser.add_argument("--saida", type=Path, default=RESULTS_DIR / "figuras" / "gradcam")
    args = parser.parse_args()

    checkpoint = WEIGHTS_DIR / "t01_cooccurrence.pt"
    if not checkpoint.exists():
        print(f"ERRO: checkpoint de T01 ausente em {checkpoint}")
        return 1

    tecnica = T01Cooccurrence().load(checkpoint)
    tecnica.model.eval()
    gradcam = GradCAM(tecnica.model)

    caminhos = list(args.imagens)
    if args.amostrar_por_gerador:
        for diretorio in sorted(p for p in (args.corpus / "fake").iterdir() if p.is_dir()):
            caminhos += sorted(diretorio.glob("*.png"))[: args.amostrar_por_gerador]
        reais = args.corpus / "real" / "coco"
        if reais.exists():
            caminhos += sorted(reais.glob("*.png"))[: args.amostrar_por_gerador]

    if not caminhos:
        print("ERRO: nenhuma imagem informada; use --imagens ou --amostrar-por-gerador")
        return 1

    args.saida.mkdir(parents=True, exist_ok=True)
    print(f"Gerando Grad-CAM para {len(caminhos)} imagens em {args.saida}\n")

    for caminho in caminhos:
        caminho = Path(caminho)
        if not caminho.exists():
            print(f"  [aviso] ausente: {caminho}")
            continue

        # A entrada e construida pelo proprio CooccurrenceDataset, e nao a mao:
        # ele aplica o reescalonamento por 1e4 que a rede espera. Alimentar a
        # matriz normalizada crua (valores da ordem de 1e-5) faz a rede emitir
        # praticamente a mesma probabilidade para qualquer imagem -- foi o que
        # ocorreu na primeira versao deste script.
        conjunto = CooccurrenceDataset([caminho], config=tecnica.config)
        entrada = conjunto[0].unsqueeze(0).to(tecnica.device)
        entrada.requires_grad_(True)

        with Image.open(caminho) as imagem:
            coocorrencia = rgb_cooccurrence_tensor(imagem, tecnica.config.offset,
                                                   tecnica.config.levels)
        mapa, probabilidade = gradcam.gerar(entrada)

        destino = args.saida / f"gradcam_{caminho.parent.name}_{caminho.stem}.png"
        figura(caminho, coocorrencia, mapa, probabilidade, destino)
        print(f"  {caminho.parent.name:<24} P(sintética)={probabilidade * 100:>5.1f}%  -> {destino.name}")

    print(f"\nFiguras em {args.saida}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
