"""Reajusta a T05 de quatro fontes na particao ``fusion`` do protocolo OOD.

O problema que isto conserta
----------------------------
``pesos/t05_fusion__quatro_fontes.pkl`` -- o modelo que a interface carrega
quando os servicos T04 estao no ar -- foi produzido por
``automacao/pipeline_t04_fusao.py``, que roda ``run_experiments.py --protocol
standard --refit-fusion`` **sem** ``--fusion-split fusion``. O padrao de
``--fusion-split`` e ``val``, que o proprio ``run_experiments.py:199`` descreve
como *"reproduz a calibracao in-distribution original"*.

Ou seja: o modelo em producao foi calibrado no regime que a secao "Por que isso
nao funcionou" de ``documentacao/DECISOES_METODOLOGICAS.md`` identifica como o que
quebra -- *"a fusao foi calibrada num cenario facil e aplicada num cenario
dificil, sem nunca ter visto como suas fontes se comportam quando erram"*. O
efeito visivel: em gerador moderno a interface responde "provavelmente real"
com tres fontes acima de 69%.

A particao ``fusion`` existe **so** no protocolo OOD -- 2.350 imagens, 1.350
reais e 1.000 do ``glide``, um gerador nao visto no treinamento. E onde a
variante B foi calibrada, e e a unica particao do projeto que mostra a fusao
como suas fontes se comportam quando erram.

O que este script NAO faz
-------------------------
**Nao avalia.** O ``run_experiments.py`` roda ``run_protocol`` logo depois do
reajuste, e no split ``test`` de ``manifesto30k_ood.csv`` (14.788 imagens) os
escores de T01/T02/T03 em cache sao de OUTRO manifesto -- os de ``clean`` tem
7.138 posicoes, do ``manifesto30k_ood_familias.csv``. E a armadilha 7. A guarda
de forma os rejeita e T02 seria recomputada: ~555 ms/imagem, mais de duas horas.

Este script para depois de gravar o modelo. A avaliacao completa e uma decisao
separada, com custo declarado.

O que ele preserva
------------------
* **Armadilha 1** -- os escores de T04 desta rodada saem sob etiqueta propria.
* **Armadilha 4** -- grava em ``pesos_t05_ood_fusion/``; nem
  ``pesos/t05_fusion.pkl`` (variante B) nem
  ``pesos/t05_fusion__quatro_fontes.pkl`` sao tocados. O modelo novo vai para
  ``pesos/`` sob **nome proprio**, e trocar o que a interface carrega e
  decisao separada.
* **Armadilha 5** -- a origem dos escores de T04 mudou, entao qualquer cache de
  ``resultados/scores/ood__T04__*`` e apagado. Os de T01/T02/T03 **nao** sao
  tocados: a origem deles nao mudou, e apaga-los custaria as duas horas acima.

Uso::

    python automacao/recalibrar_t05_fusion.py
    python automacao/recalibrar_t05_fusion.py --so-fundir   # so o passo 1
"""

from __future__ import annotations

import argparse
import csv
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

PYTHON = sys.executable
PESOS_VARIANTE = ROOT / "pesos_t05_ood_fusion"
MANIFESTO = ROOT / "data" / "manifesto30k_ood.csv"
ESCORES_FUNDIDOS = ROOT / "resultados" / "t04_escores_componentes_ood_completo.csv"
COMPONENTES = ("object_shadow", "perspective_fields", "line_segment")

# Ordem importa: a primeira fonte que traz um stem vence. As duas primeiras
# produziram numeros ja reportados -- reescrever um escore delas mudaria em
# silencio um valor que esta no capitulo.
FONTES_ESCORES = (
    ROOT / "resultados" / "t04_escores_componentes.csv",              # dissertacao
    ROOT / "resultados" / "t04_escores_componentes_ood.csv",          # benchmark
    ROOT / "resultados" / "t04_escores_componentes_ood_fusion.csv",   # esta rodada
)


# ----------------------------------------------------------------------
# Passo 1 -- um arquivo de escores que cubra o protocolo OOD inteiro
# ----------------------------------------------------------------------

def fundir_escores() -> int:
    """Junta as tres rodadas de T04 num arquivo unico, sem sobrescrever nada.

    A regra e **quem chegou primeiro vence**. As rodadas da dissertacao e do
    benchmark ja produziram numeros reportados; se esta rodada reextraiu uma
    imagem que elas ja cobriam -- e reextraiu, as 1.350 reais da particao
    ``fusion`` --, o escore que prevalece e o antigo. Assim a fusao passa a ver
    1.000 imagens novas e **nenhum** valor antigo alterado.
    """
    if not MANIFESTO.exists():
        print(f"ERRO: manifesto ausente em {MANIFESTO}")
        return 1

    faltando = [p for p in FONTES_ESCORES if not p.exists()]
    if faltando:
        print("ERRO: fontes de escore ausentes:")
        for p in faltando:
            print(f"  {p}")
        print("\nRode antes: automacao\\t04_glide_fusao.bat")
        print("e depois  : python automacao/consolidar_escores_t04.py "
              "--conjunto tcc3_ood_fusion --sufixo _ood_fusion "
              "--saida resultados/t04_escores_componentes_ood_fusion.csv")
        return 1

    # Split verdadeiro de cada imagem, segundo o manifesto OOD. As tres fontes
    # trazem a coluna `split` do manifesto delas, que nao e este.
    split_por_stem: dict[str, str] = {}
    rotulo_por_stem: dict[str, str] = {}
    for registro in csv.DictReader(MANIFESTO.open(newline="", encoding="utf-8")):
        stem = Path(registro["path"]).stem
        split_por_stem[stem] = registro["split"]
        rotulo_por_stem[stem] = registro["label"]

    fundido: dict[str, dict[str, str]] = {}
    origem: dict[str, Path] = {}
    for caminho in FONTES_ESCORES:
        novos = 0
        for linha in csv.DictReader(caminho.open(newline="", encoding="utf-8")):
            stem = linha["arquivo"]
            if stem in fundido:          # quem chegou primeiro vence
                continue
            fundido[stem] = {c: (linha.get(c) or "").strip() for c in COMPONENTES}
            origem[stem] = caminho
            novos += 1
        print(f"[fundir] {caminho.name:<40} +{novos:>6} stems")

    # Cobertura. O agregador de T04 falha alto para imagem fora do arquivo,
    # e falhar aqui e muito mais barato que falhar no meio do reajuste.
    #
    # A exigencia dura vale para a particao `fusion`, a unica que o reajuste
    # consome. Para as demais a ausencia e relatada, nao fatal: ha falhas
    # legitimas e pre-existentes do extrator objeto-sombra, registradas em
    # data/t04_object_shadow/*_falhas_*.csv desde a rodada da dissertacao --
    # coco_022232 (KeyError na deteccao) e uma delas, no split `train`.
    # Exigir 100% do manifesto inteiro faria uma falha antiga e conhecida
    # bloquear uma operacao que nem toca naquele split.
    exigidos = {s for s, sp in split_por_stem.items() if sp == "fusion"}
    ausentes = sorted(exigidos - set(fundido))
    if ausentes:
        print(f"\nERRO: {len(ausentes)} imagens da particao 'fusion' sem "
              "escore de T04 -- e esta e a particao do reajuste.")
        print(f"  exemplos: {ausentes[:5]}")
        return 1
    print("[fundir] particao 'fusion': cobertura 100%")

    for split in sorted(set(split_por_stem.values())):
        if split == "fusion":
            continue
        do_split = {s for s, sp in split_por_stem.items() if sp == split}
        faltam = do_split - set(fundido)
        if faltam:
            print(f"[fundir] AVISO {split}: {len(faltam)} sem escore "
                  f"(ex.: {sorted(faltam)[:3]}) -- falha pre-existente do "
                  "extrator, nao bloqueia este reajuste")

    linhas = []
    for stem in sorted(s for s in split_por_stem if s in fundido):
        linha = {"arquivo": stem, "split": split_por_stem[stem],
                 "label": rotulo_por_stem[stem]}
        linha.update(fundido[stem])
        linhas.append(linha)

    ESCORES_FUNDIDOS.parent.mkdir(parents=True, exist_ok=True)
    with ESCORES_FUNDIDOS.open("w", newline="", encoding="utf-8") as fh:
        escritor = csv.DictWriter(
            fh, fieldnames=["arquivo", "split", "label", *COMPONENTES])
        escritor.writeheader()
        escritor.writerows(linhas)

    print(f"\n[fundir] {len(linhas)} imagens em {ESCORES_FUNDIDOS}")
    por_split: dict[str, int] = {}
    for linha in linhas:
        por_split[linha["split"]] = por_split.get(linha["split"], 0) + 1
    print(f"[fundir] por split: {por_split}")
    return 0


# ----------------------------------------------------------------------
# Passo 2 -- ambiente isolado, para nao tocar em modelo reportado
# ----------------------------------------------------------------------

def preparar_pesos() -> dict[str, str]:
    from codigo.configuracao import WEIGHTS_DIR

    PESOS_VARIANTE.mkdir(parents=True, exist_ok=True)
    for nome in ("t01_cooccurrence.pt", "t03_benford.pkl"):
        origem, destino = WEIGHTS_DIR / nome, PESOS_VARIANTE / nome
        if origem.exists() and not destino.exists():
            shutil.copy2(origem, destino)
            print(f"[pesos] copia de {nome}")

    origem_spai, destino_spai = WEIGHTS_DIR / "spai.pth", PESOS_VARIANTE / "spai.pth"
    if origem_spai.exists() and not destino_spai.exists():
        try:
            os.link(origem_spai, destino_spai)      # 891 MB; link em vez de copia
            print("[pesos] link para spai.pth")
        except OSError:
            shutil.copy2(origem_spai, destino_spai)
            print("[pesos] copia de spai.pth")

    geometria = WEIGHTS_DIR / "projective_geometry"
    if geometria.exists() and not (PESOS_VARIANTE / "projective_geometry").exists():
        shutil.copytree(geometria, PESOS_VARIANTE / "projective_geometry")
        print("[pesos] copia de projective_geometry/")

    return {"TCC3_WEIGHTS_DIR": str(PESOS_VARIANTE)}


def limpar_cache_t04() -> None:
    """Armadilha 5: a origem dos escores de T04 mudou.

    Apaga **so** o cache de T04. O de T01/T02/T03 continua valido -- a origem
    deles nao mudou -- e recomputa-lo custaria mais de duas horas de T02.
    """
    cache = ROOT / "resultados" / "scores"
    apagados = [p for p in cache.glob("ood__T04__*.npy")]
    for p in apagados:
        p.unlink()
    print(f"[cache] {len(apagados)} arquivos de ood__T04__* apagados")


# ----------------------------------------------------------------------
# Passo 3 -- o reajuste, pelo mesmo caminho de codigo do run_experiments
# ----------------------------------------------------------------------

def reajustar() -> int:
    """Executado no processo filho, ja com o ambiente aplicado."""
    from codigo.configuracao import WEIGHTS_DIR
    from codigo.experimentos.runner import ExperimentRunner
    from automacao.run_experiments import build_techniques, load_models

    device = "cpu"
    try:
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        pass

    print(f"Dispositivo: {device}")
    print(f"Pesos      : {WEIGHTS_DIR}")
    print(f"Manifesto  : {MANIFESTO}\n")

    tecnicas = build_techniques(["T01", "T02", "T03", "T04", "T05"], device)
    print(f"Tecnicas registradas: {sorted(tecnicas)}\n")

    if not load_models({k: v for k, v in tecnicas.items() if k != "T05"}):
        print("ERRO: modelos treinados nao encontrados")
        return 1

    runner = ExperimentRunner(MANIFESTO, tecnicas, cache_scores=True)

    print("--- Reajuste de T05 sobre a particao 'fusion' do protocolo OOD ---")
    erro = runner.fit_fusion(protocol="ood", split="fusion")
    if erro is not None:
        print(f"  T05: FALHOU ({erro})")
        return 1
    print("  T05: ok")

    fusao = tecnicas["T05"]
    destino = WEIGHTS_DIR / "t05_fusion.pkl"
    fusao.save(destino)
    print(f"  pesos: {destino}")

    print("\n  peso por dominio de evidencia:")
    for nome, valor in fusao.contribution_weights().items():
        print(f"    {nome}: {valor:+.4f}")
    return 0


def preservar() -> None:
    """Leva o modelo para ``pesos/`` sob nome proprio.

    **Nao** sobrescreve o que a interface carrega hoje. Trocar o modelo em
    producao e decisao separada, e reversivel: basta apontar o
    ``_load_all()`` para o outro arquivo.
    """
    from codigo.configuracao import WEIGHTS_DIR

    origem = PESOS_VARIANTE / "t05_fusion.pkl"
    if not origem.exists():
        print(f"[aviso] modelo nao encontrado em {origem}")
        return
    destino = WEIGHTS_DIR / "t05_fusion__quatro_fontes_ood.pkl"
    shutil.copy2(origem, destino)
    print(f"\n[pesos] modelo recalibrado em {destino}")
    print(f"[pesos] {WEIGHTS_DIR / 't05_fusion__quatro_fontes.pkl'} NAO foi tocado")
    print(f"[pesos] {WEIGHTS_DIR / 't05_fusion.pkl'} (variante B) NAO foi tocado")


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--so-fundir", action="store_true",
                        help="para depois de fundir os escores de T04")
    parser.add_argument("--interno", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()

    if args.interno:
        return reajustar()

    if (codigo := fundir_escores()) != 0:
        return codigo
    if args.so_fundir:
        return 0

    ambiente = preparar_pesos()
    ambiente["TCC3_T04_ESCORES"] = str(ESCORES_FUNDIDOS)
    limpar_cache_t04()

    print(f"\n[T04] escores: {ESCORES_FUNDIDOS}")
    print(f"[pesos] esta variante grava em {PESOS_VARIANTE}\n")

    # Subprocesso porque TCC3_WEIGHTS_DIR e TCC3_T04_ESCORES sao lidos na
    # importacao dos modulos, nao em tempo de chamada.
    concluido = subprocess.run(
        [PYTHON, "-u", str(Path(__file__).resolve()), "--interno"],
        cwd=str(ROOT), check=False, env={**os.environ, **ambiente})
    if concluido.returncode == 0:
        preservar()
    return concluido.returncode


if __name__ == "__main__":
    raise SystemExit(main())
