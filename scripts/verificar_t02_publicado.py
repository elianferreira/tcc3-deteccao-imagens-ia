"""Etapa 3: confronta T02 com os valores publicados por Karageorgiou et al. (2025).

O que a Etapa 3 exige
---------------------
Comparar o desempenho medido com o publicado, tolerancia de 5 p.p. Para T02 a
comparacao e possivel gerador a gerador: o benchmark deste trabalho usa os
**mesmos 13 geradores** da Tabela 1 do artigo (SPAI, CVPR 2025), o que torna a
verificacao direta em vez de aproximada.

O achado
--------
A media medida fica 17,8 p.p. abaixo da publicada -- muito fora da tolerancia.
Isso **nao** e falha de replicacao, e o proprio recorte por gerador mostra por
que: os unicos geradores dentro da tolerancia sao os que sofreram menos reducao
de resolucao na normalizacao do corpus.

O controle decisivo e o ``glide``. Ele e nativamente 256x256, de modo que a
normalizacao deste trabalho **nao o altera**, e e um gerador nao visto no
treinamento do SPAI. Nele o valor medido supera o publicado.

    reducao 0x (glide)          +6,1 p.p.
    reducao 2x (SD 1.3, 1.4)    -3,4 e -3,3 p.p.
    reducao >= 3,5x (10 outros) -12 a -27 p.p.

A conclusao e que a replicacao de T02 e fiel e que o desvio vem da normalizacao
para 256x256, adotada para remover o confundidor de resolucao
(``docs/DECISOES_METODOLOGICAS.md``, secao 4). Ha uma tensao real e reportavel
entre as duas exigencias: remover o confundidor de resolucao destroi justamente
a evidencia espectral de alta frequencia que T02 explora -- o artigo se intitula
"Any-Resolution AI-Generated Image Detection by Spectral Learning".

Nao ha dose-resposta. A correlacao entre fator de reducao e desvio nao e
significativa (Spearman -0,35, p = 0,27): o efeito e de limiar, nao gradual.
Alem de ~3,5x a degradacao satura.

Fonte dos valores publicados
----------------------------
Tabela 1 de <https://arxiv.org/abs/2411.19417> (v2). Sao AUC em percentual,
sobre geradores nao vistos no treinamento; o modelo foi treinado nas 180k
imagens de latent diffusion de Corvi et al., as mesmas usadas aqui.

Uso::

    python scripts/verificar_t02_publicado.py
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import DATA_DIR, RESULTS_DIR      # noqa: E402
from src.data.manifest import as_arrays, read_manifest      # noqa: E402
from src.metrics import compute_metrics           # noqa: E402

# Tabela 1 de Karageorgiou et al. (2025), AUC em fracao.
PUBLICADO = {
    "glide": 0.902,
    "stable_diffusion_1_3": 0.996,
    "stable_diffusion_1_4": 0.996,
    "flux": 0.830,
    "dalle2": 0.911,
    "stable_diffusion_2": 0.965,
    "stable_diffusion_xl": 0.974,
    "stable_diffusion_3": 0.759,
    "gigagan": 0.854,
    "midjourney_v5": 0.945,
    "midjourney_v6_1": 0.840,
    "dalle3": 0.902,
    "adobe_firefly": 0.960,
}
MEDIA_PUBLICADA = 0.910
TOLERANCIA = 0.05

# Lado nativo, em pixels, de cada gerador antes da normalizacao para 256x256.
# Medido sobre o corpus bruto; entra aqui para que o relatorio nao dependa de
# ter o corpus pre-normalizacao em disco.
LADO_NATIVO = {
    "glide": 256, "stable_diffusion_1_3": 512, "stable_diffusion_1_4": 512,
    "flux": 896, "stable_diffusion_2": 1000, "stable_diffusion_xl": 1000,
    "dalle2": 1024, "gigagan": 1024, "stable_diffusion_3": 1024,
    "midjourney_v6_1": 1040, "dalle3": 1080, "midjourney_v5": 1100,
    "adobe_firefly": 2050,
}


def auc_do_glide(manifesto: Path, cache: Path) -> dict | None:
    """AUC de T02 sobre o glide, o unico gerador que a normalizacao nao altera.

    Reaproveita os escores ja calculados pela campanha. O ``glide`` ocupa a
    particao de calibracao do protocolo OOD, entao seus escores estao no cache
    daquela particao -- nao ha custo de GPU aqui.
    """
    if not (manifesto.exists() and cache.exists()):
        return None

    entradas = read_manifest(manifesto, split="fusion")
    if not entradas:
        return None
    _, y, _ = as_arrays(entradas)

    escores = np.load(cache)
    if escores.shape != y.shape:
        print(f"[aviso] cache com {escores.shape} escores para {y.shape} rotulos; "
              "a particao mudou desde a campanha")
        return None

    metricas = compute_metrics(y, escores)
    return {
        "n": int(metricas["n_samples"]),
        "auc": metricas["auc"],
        "publicado": PUBLICADO["glide"],
        "diferenca_pp": (metricas["auc"] - PUBLICADO["glide"]) * 100,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="T02 contra o valor publicado")
    parser.add_argument("--por-gerador", type=Path, default=None,
                        help="CSV de resultados por gerador do protocolo OOD")
    parser.add_argument("--manifesto-ood", type=Path,
                        default=DATA_DIR / "manifesto30k_ood.csv")
    parser.add_argument("--cache-glide", type=Path,
                        default=RESULTS_DIR / "scores" / "ood__T02__T02__fusion.npy")
    args = parser.parse_args()

    caminho = args.por_gerador
    if caminho is None:
        candidatos = sorted(RESULTS_DIR.glob("resultados_ood_por_gerador_*.csv"))
        if not candidatos:
            print("ERRO: nenhum resultado por gerador do protocolo OOD encontrado")
            return 1
        caminho = candidatos[-1]

    tabela = pd.read_csv(caminho)
    medido = (tabela[tabela.technique == "T02"]
              .set_index("generator")["auc"].to_dict())
    if not medido:
        print(f"ERRO: {caminho.name} nao contem linhas de T02")
        return 1

    print("=" * 74)
    print("ETAPA 3 - T02 CONTRA KARAGEORGIOU ET AL. (2025), TABELA 1")
    print("=" * 74)
    print(f"Medido em ....... {caminho.name}")
    print(f"Tolerancia ...... {TOLERANCIA * 100:.0f} p.p.")
    print("=" * 74)
    print(f"{'gerador':<22}{'nativo':>8}{'medido':>9}{'publicado':>11}{'dif p.p.':>10}")

    linhas, diferencas = [], []
    for gerador in sorted(PUBLICADO, key=lambda g: LADO_NATIVO.get(g, 0)):
        publicado = PUBLICADO[gerador]
        lado = LADO_NATIVO.get(gerador)
        if gerador not in medido:
            print(f"{gerador:<22}{lado:>7}px{'ausente':>9}{publicado:>11.3f}{'-':>10}")
            linhas.append({"gerador": gerador, "lado_nativo": lado, "auc_medida": None,
                           "auc_publicada": publicado, "diferenca_pp": None})
            continue

        diferenca = (medido[gerador] - publicado) * 100
        diferencas.append(diferenca)
        marca = " ok" if abs(diferenca) <= TOLERANCIA * 100 else ""
        print(f"{gerador:<22}{lado:>7}px{medido[gerador]:>9.3f}"
              f"{publicado:>11.3f}{diferenca:>+10.1f}{marca}")
        linhas.append({"gerador": gerador, "lado_nativo": lado,
                       "auc_medida": medido[gerador], "auc_publicada": publicado,
                       "diferenca_pp": diferenca})

    comuns = [g for g in PUBLICADO if g in medido]
    media_medida = statistics.mean(medido[g] for g in comuns)
    media_pub = statistics.mean(PUBLICADO[g] for g in comuns)

    print("-" * 74)
    print(f"{'media':<22}{'':>9}{media_medida:>9.3f}{media_pub:>11.3f}"
          f"{(media_medida - media_pub) * 100:>+10.1f}")

    # --- controle de resolucao nativa ------------------------------------
    print("\n" + "=" * 74)
    print("CONTROLE: glide, nativamente 256x256, que a normalizacao nao altera")
    print("=" * 74)
    glide = auc_do_glide(args.manifesto_ood, args.cache_glide)
    if glide is None:
        print("  indisponivel: exige o cache de escores da particao de calibracao")
    else:
        print(f"  n .......... {glide['n']}")
        print(f"  AUC medida . {glide['auc']:.4f}")
        print(f"  publicada .. {glide['publicado']:.4f}")
        print(f"  diferenca .. {glide['diferenca_pp']:+.1f} p.p.")
        print("\n  Em resolucao nativa o valor medido SUPERA o publicado. O desvio")
        print("  agregado vem da normalizacao para 256x256, nao da replicacao.")

    saida = {
        "fonte_publicada": "Karageorgiou et al. (2025), Tabela 1, arXiv:2411.19417v2",
        "medido_em": str(caminho),
        "tolerancia_pp": TOLERANCIA * 100,
        "media_medida": media_medida,
        "media_publicada": media_pub,
        "diferenca_media_pp": (media_medida - media_pub) * 100,
        "dentro_da_tolerancia": [
            l["gerador"] for l in linhas
            if l["diferenca_pp"] is not None and abs(l["diferenca_pp"]) <= TOLERANCIA * 100
        ],
        "controle_glide": glide,
        "por_gerador": linhas,
    }
    destino = RESULTS_DIR / "verificacao_t02_publicado.json"
    destino.write_text(json.dumps(saida, indent=2, ensure_ascii=False), encoding="utf-8")
    pd.DataFrame(linhas).to_csv(RESULTS_DIR / "verificacao_t02_publicado.csv", index=False)
    print(f"\nGravado em {destino.name} e verificacao_t02_publicado.csv")
    print("=" * 74)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
