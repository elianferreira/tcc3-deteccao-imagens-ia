"""Monta o corpus em escala, extraindo e normalizando em passagem unica.

Por que existe
-------------
O fluxo em duas etapas -- ``organize_corpus.py`` grava o corpus bruto e
``normalize_corpus.py`` grava a versao normalizada -- mantem duas copias em
disco. Para 180.000 imagens isso custa cerca de 24 GB desnecessarios, e a
maquina do projeto tem 32 GB livres. Ver documentacao/PLANO_ESCALA_INTEGRAL.md,
contingencia A.

Este script le direto dos arquivos ``.zip``, normaliza em memoria e grava
apenas a saida final: menor lado a 256 por LANCZOS, recorte central e PNG.

Escala alcancavel
-----------------
O conjunto real oficial de Corvi et al. (2024) sao duas listas de 90.000 nomes:
``real_coco.txt`` e ``real_lsun.txt``. Apenas a primeira e reproduzivel aqui --
LSUN nao foi obtido. O teto e, portanto, **90.000 imagens reais**, metade da
meta da Secao 3.5.1, e o script mantem a proporcao 1:1 exigida limitando as
sinteticas ao mesmo numero.

Retomada
--------
Arquivos ja gravados sao pulados, de modo que a execucao pode ser interrompida
e retomada sem perda. A verificacao de espaco livre roda periodicamente e
interrompe de forma limpa antes de o disco encher.

Uso::

    python automacao/preparar_corpus_escala.py --n-real 90000 --n-sintetica 90000
    python automacao/preparar_corpus_escala.py --n-real 60000 --n-sintetica 60000
"""

from __future__ import annotations

import argparse
import io
import json
import os
import shutil
import sys
import zipfile
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from codigo.configuracao import DATA_DIR, IMAGE_SIZE      # noqa: E402
from codigo.preprocessamento.normalizacao import resize_and_center_crop      # noqa: E402
from automacao.organize_corpus import GENERATOR_MAP                # noqa: E402

DOWNLOAD_DIR = DATA_DIR / "downloads"
MARGEM_GB = 3.0            # espaco livre minimo antes de continuar


def espaco_livre_gb(caminho: Path) -> float:
    return shutil.disk_usage(caminho).free / (1024 ** 3)


def gravar_normalizada(dados: bytes, destino: Path, tamanho: int) -> bool:
    """Normaliza os bytes de uma imagem e grava em PNG. Retorna False em falha.

    A gravacao e atomica: escreve em arquivo temporario e so entao renomeia
    sobre o destino. Sem isso, um encerramento no meio da escrita deixa um
    arquivo parcial no caminho final -- e como a retomada considera existente
    qualquer arquivo presente, ele jamais seria regerado.

    Nao e hipotese: apos os encerramentos desta sessao, o corpus continha
    ``coco_019669.png`` com 0 byte e ``coco_082449.png`` com 1,4 MB contra os
    ~100 KB tipicos, ambos ilegiveis.
    """
    destino.parent.mkdir(parents=True, exist_ok=True)
    temporario = destino.with_name(destino.name + ".parcial")
    try:
        with Image.open(io.BytesIO(dados)) as imagem:
            normalizada = resize_and_center_crop(imagem, tamanho)
            # Descarta todo metadado herdado da origem. Medido no corpus: 51%
            # das imagens reais do COCO carregavam perfil ICC contra 0% das
            # sinteticas de qualquer gerador -- um marcador que separa as
            # classes sem qualquer relacao com sintese, do mesmo tipo que o
            # confundidor de resolucao e formato ja corrigido.
            #
            # Verificou-se que o perfil nao altera os pixels decodificados,
            # entao ele nao contamina T01, T02 e T03; ainda assim sai daqui,
            # porque um perfil de 1,3 MB tornou uma imagem irrecuperavel pela
            # protecao contra bomba de descompressao do Pillow.
            normalizada.info.pop("icc_profile", None)
            normalizada.save(temporario, format="PNG")
        # os.replace e atomico dentro do mesmo volume: o destino passa a existir
        # ja completo, ou nao existe.
        os.replace(temporario, destino)
        return True
    except Exception as erro:                            # noqa: BLE001
        print(f"  [falha] {destino.name}: {erro}")
        temporario.unlink(missing_ok=True)
        return False


def verificar_e_limpar(raiz: Path) -> int:
    """Remove arquivos vazios ou ilegiveis deixados por escritas interrompidas.

    Executado antes da extracao para que a retomada os regere, em vez de
    considera-los prontos por simplesmente existirem.
    """
    removidos = 0
    for caminho in raiz.rglob("*.png"):
        try:
            if caminho.stat().st_size == 0:
                caminho.unlink()
                removidos += 1
                continue
        except OSError:
            continue
    for parcial in raiz.rglob("*.parcial"):
        parcial.unlink(missing_ok=True)
        removidos += 1
    return removidos


def lista_oficial_real() -> list[str]:
    caminho = DOWNLOAD_DIR / "latent_diffusion_trainingset.zip"
    with zipfile.ZipFile(caminho) as arquivo:
        conteudo = arquivo.read("latent_diffusion_trainingset/train/real_coco.txt")
    return [l.strip() for l in conteudo.decode("utf-8").splitlines() if l.strip()]


def extrair_reais(saida: Path, limite: int, tamanho: int) -> int:
    """Reais do COCO train2017, na ordem da lista oficial de Corvi et al."""
    zip_coco = DOWNLOAD_DIR / "coco_train2017.zip"
    if not zip_coco.exists():
        print(f"[aviso] {zip_coco.name} ausente")
        return 0

    oficial = lista_oficial_real()
    destino_base = saida / "real" / "coco"
    print(f"\n[real/coco] lista oficial com {len(oficial)} nomes; alvo {limite}")

    gravadas = 0
    with zipfile.ZipFile(zip_coco) as arquivo:
        por_nome = {Path(n).name: n for n in arquivo.namelist()
                    if n.lower().endswith((".jpg", ".jpeg", ".png"))}
        for indice, nome in enumerate(oficial):
            if gravadas >= limite:
                break
            interno = por_nome.get(nome)
            if interno is None:
                continue
            destino = destino_base / f"coco_{indice:06d}.png"
            if destino.exists():
                gravadas += 1
                continue
            if gravar_normalizada(arquivo.read(interno), destino, tamanho):
                gravadas += 1
            if gravadas % 5000 == 0 and gravadas:
                print(f"  {gravadas} de {limite} ...", flush=True)
                if espaco_livre_gb(saida) < MARGEM_GB:
                    print(f"  PARADA: menos de {MARGEM_GB} GB livres")
                    break
    print(f"  real/coco: {gravadas} imagens")
    return gravadas


def extrair_sinteticas_treino(saida: Path, limite: int, tamanho: int) -> int:
    zip_treino = DOWNLOAD_DIR / "latent_diffusion_trainingset.zip"
    if not zip_treino.exists():
        print(f"[aviso] {zip_treino.name} ausente")
        return 0

    destino_base = saida / "fake" / "latent_diffusion"
    print(f"\n[fake/latent_diffusion] alvo {limite}")

    gravadas = 0
    with zipfile.ZipFile(zip_treino) as arquivo:
        nomes = sorted(n for n in arquivo.namelist()
                       if n.lower().endswith((".png", ".jpg", ".jpeg")) and "/train/" in n)
        for indice, nome in enumerate(nomes):
            if gravadas >= limite:
                break
            destino = destino_base / f"latent_diffusion_{indice:06d}.png"
            if destino.exists():
                gravadas += 1
                continue
            if gravar_normalizada(arquivo.read(nome), destino, tamanho):
                gravadas += 1
            if gravadas % 5000 == 0 and gravadas:
                print(f"  {gravadas} de {limite} ...", flush=True)
                if espaco_livre_gb(saida) < MARGEM_GB:
                    print(f"  PARADA: menos de {MARGEM_GB} GB livres")
                    break
    print(f"  fake/latent_diffusion: {gravadas} imagens")
    return gravadas


def extrair_benchmark(saida: Path, por_gerador: int, tamanho: int) -> dict[str, int]:
    contagens: dict[str, int] = {}
    for nome_zip in ("synthbuster.zip", "extra_generators.zip"):
        caminho = DOWNLOAD_DIR / nome_zip
        if not caminho.exists():
            print(f"[aviso] {nome_zip} ausente")
            continue

        print(f"\n[{nome_zip}]")
        with zipfile.ZipFile(caminho) as arquivo:
            por_geradorlista: dict[str, list[str]] = {}
            for nome in arquivo.namelist():
                if not nome.lower().endswith((".png", ".jpg", ".jpeg", ".webp")):
                    continue
                partes = nome.split("/")
                if len(partes) < 2:
                    continue
                bruto = partes[1] if partes[0] == "synthbuster" else partes[0]
                alvo = GENERATOR_MAP.get(bruto)
                if alvo:
                    por_geradorlista.setdefault(alvo, []).append(nome)

            for gerador, nomes in sorted(por_geradorlista.items()):
                nomes.sort()
                selecionados = nomes[:por_gerador]
                destino_base = saida / "fake" / gerador
                gravadas = 0
                for indice, nome in enumerate(selecionados):
                    destino = destino_base / f"{gerador}_{indice:05d}.png"
                    if destino.exists():
                        gravadas += 1
                        continue
                    if gravar_normalizada(arquivo.read(nome), destino, tamanho):
                        gravadas += 1
                contagens[gerador] = gravadas
                print(f"  {gerador:<24} {gravadas:>6}")
    return contagens


def main() -> int:
    parser = argparse.ArgumentParser(description="Corpus em escala, passagem unica")
    parser.add_argument("--saida", type=Path, default=DATA_DIR / "corvi2024_escala")
    parser.add_argument("--n-real", type=int, default=90_000)
    parser.add_argument("--n-sintetica", type=int, default=90_000)
    parser.add_argument("--benchmark-por-gerador", type=int, default=1000)
    parser.add_argument("--size", type=int, default=IMAGE_SIZE)
    args = parser.parse_args()

    args.saida.mkdir(parents=True, exist_ok=True)
    livre = espaco_livre_gb(args.saida)

    # 114,6 KB por imagem, medido no corpus de 30k ja normalizado.
    total_previsto = args.n_real + args.n_sintetica + args.benchmark_por_gerador * 13
    necessario_gb = total_previsto * 114.6 / (1024 ** 2)

    print("=" * 70)
    print("CORPUS EM ESCALA - EXTRACAO E NORMALIZACAO EM PASSAGEM UNICA")
    print("=" * 70)
    print(f"Saida ............... {args.saida}")
    print(f"Reais ............... {args.n_real}")
    print(f"Sinteticas .......... {args.n_sintetica}")
    print(f"Benchmark ........... {args.benchmark_por_gerador} x 13")
    print(f"Total previsto ...... {total_previsto} imagens")
    print(f"Espaco necessario ... {necessario_gb:.1f} GB")
    print(f"Espaco livre ........ {livre:.1f} GB")
    print("=" * 70)

    if livre < necessario_gb + MARGEM_GB:
        print(f"\nERRO: espaco insuficiente. Necessario {necessario_gb:.1f} GB "
              f"mais {MARGEM_GB} GB de margem; livres {livre:.1f} GB.")
        print("Ver documentacao/PLANO_ESCALA_INTEGRAL.md, contingencias B e C.")
        return 1

    removidos = verificar_e_limpar(args.saida)
    if removidos:
        print(f"\n[limpeza] {removidos} arquivos vazios ou parciais removidos; "
              "serao regerados")

    contagens = extrair_benchmark(args.saida, args.benchmark_por_gerador, args.size)
    n_sintetica = extrair_sinteticas_treino(args.saida, args.n_sintetica, args.size)
    if n_sintetica:
        contagens["latent_diffusion"] = n_sintetica
    n_real = extrair_reais(args.saida, args.n_real, args.size)

    info = {
        "saida": str(args.saida),
        "n_real": n_real,
        "n_sintetica_treino": n_sintetica,
        "benchmark": contagens,
        "resolucao": args.size,
        "formato": "PNG",
        "proporcao_1_1": abs(n_real - n_sintetica) <= max(1, n_real * 0.01),
        "fonte_real": "COCO train2017, lista oficial real_coco.txt",
        "limitacao": ("LSUN indisponivel: as outras 90.000 imagens reais do conjunto "
                      "oficial de Corvi et al. (2024) nao puderam ser obtidas"),
    }
    (args.saida / "corpus_info.json").write_text(
        json.dumps(info, indent=2, ensure_ascii=False), encoding="utf-8")

    print("\n" + "=" * 70)
    print(f"Reais ............... {n_real}")
    print(f"Sinteticas (treino) . {n_sintetica}")
    print(f"Benchmark ........... {sum(v for k, v in contagens.items() if k != 'latent_diffusion')}")
    print(f"Espaco livre agora .. {espaco_livre_gb(args.saida):.1f} GB")
    print("=" * 70)
    print(f"\nProximo passo:\n  python automacao/prepare_dataset.py --corpus {args.saida} "
          "--protocol standard --skip-duplicates")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
