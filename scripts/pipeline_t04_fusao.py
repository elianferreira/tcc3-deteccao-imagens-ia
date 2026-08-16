"""Reexecuta o protocolo padrao com T04 como quarta fonte da fusao T05.

Por que um script e nao uma linha de comando
--------------------------------------------
A execucao precisa de duas garantias que se erram facilmente na mao:

1. **Nao retreinar nada.** Sem ``--fit``, T01 e T03 sao carregados dos pesos ja
   ajustados e os escores de T01, T02 e T03 vem do cache em disco
   (``results/scores/standard__*``). O unico escore novo e o de T04, lido dos
   valores pre-extraidos. Retreinar aqui destruiria a comparabilidade com os
   resultados ja documentados.

2. **Nao sobrescrever o T05 do protocolo padrao.** ``--refit-fusion`` reajusta a
   fusao -- necessario, porque um T05 ajustado com tres fontes aprendeu a
   decidir sem a quarta e continuaria ignorando-a. Mas o arquivo resultante nao
   pode substituir ``weights/t05_fusion.pkl``, que produziu a AUC 0,9996
   documentada e e o que a interface carrega.

A segunda garantia usa ``TCC3_WEIGHTS_DIR``, previsto em ``src/config.py``,
exatamente como faz ``scripts/pipeline_pos_multiseed.py``: esta variante grava
em um diretorio proprio, com os modelos pequenos replicados e o checkpoint de
891 MB do SPAI apenas referenciado por link.

O que se espera do resultado
----------------------------
T04 (objeto-sombra) obteve AUC 0,5383 isolada -- praticamente o acaso, ver
secao 4.8 de ``docs/RESULTADOS.md``. A pergunta que esta execucao responde nao e
se a fusao melhora, e sim se **piora**: uma fonte sem sinal pode degradar a
decisao se o classificador de fusao lhe atribuir peso. O peso aprendido fica em
``contribution_weights()`` e e o numero a reportar.

Uso::

    python scripts/pipeline_t04_fusao.py
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import WEIGHTS_DIR      # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
PYTHON = sys.executable
PESOS_VARIANTE = ROOT / "weights_t04_fusao"

# Modelos pequenos que a variante precisa carregar; copiados para o diretorio
# proprio. T02 e T04 nao sao treinados e nao aparecem aqui.
MODELOS_PEQUENOS = ("t01_cooccurrence.pt", "t03_benford.pkl")


def preparar_pesos() -> dict[str, str]:
    """Diretorio de pesos separado, para preservar o protocolo padrao."""
    PESOS_VARIANTE.mkdir(parents=True, exist_ok=True)

    for nome in MODELOS_PEQUENOS:
        origem = WEIGHTS_DIR / nome
        destino = PESOS_VARIANTE / nome
        if origem.exists() and not destino.exists():
            shutil.copy2(origem, destino)
            print(f"[pesos] copia de {nome}")

    # 891 MB; link em vez de copia.
    origem_spai = WEIGHTS_DIR / "spai.pth"
    destino_spai = PESOS_VARIANTE / "spai.pth"
    if origem_spai.exists() and not destino_spai.exists():
        try:
            os.link(origem_spai, destino_spai)
            print("[pesos] link para spai.pth")
        except OSError:
            shutil.copy2(origem_spai, destino_spai)
            print("[pesos] copia de spai.pth")

    geometria = WEIGHTS_DIR / "projective_geometry"
    if geometria.exists() and not (PESOS_VARIANTE / "projective_geometry").exists():
        shutil.copytree(geometria, PESOS_VARIANTE / "projective_geometry")
        print("[pesos] copia de projective_geometry/")

    return {"TCC3_WEIGHTS_DIR": str(PESOS_VARIANTE)}


def main() -> int:
    parser = argparse.ArgumentParser(description="Protocolo padrao com T04 na fusao")
    parser.add_argument("--manifesto", type=Path,
                        default=ROOT / "data" / "manifesto30k_standard.csv")
    parser.add_argument("--escores-t04", type=Path,
                        default=ROOT / "results" / "t04_escores_combined.csv")
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    if not args.escores_t04.exists():
        print(f"ERRO: escores de T04 ausentes em {args.escores_t04}")
        print("Execute antes: python scripts/avaliar_t04_corpus.py")
        return 1

    env = preparar_pesos()
    # Liga o modo de escores pre-extraidos de T04. E deliberadamente explicito:
    # fora da campanha -- na interface, por exemplo -- T04 deve continuar
    # reportando indisponibilidade, porque os escores cobrem apenas o corpus
    # deste trabalho e nao imagens arbitrarias.
    env["TCC3_T04_ESCORES"] = str(args.escores_t04)

    print(f"\n[pesos] esta variante grava em {PESOS_VARIANTE}")
    print("[pesos] weights/t05_fusion.pkl permanece intacto")
    print(f"[T04] escores pre-extraidos: {args.escores_t04}\n")

    comando = [
        PYTHON, "-u", str(ROOT / "scripts" / "run_experiments.py"),
        "--protocol", "standard",
        "--manifest", str(args.manifesto),
        "--techniques", "T01", "T02", "T03", "T04", "T05",
        "--refit-fusion",
    ]
    if args.device:
        comando += ["--device", args.device]

    print(" ".join(comando) + "\n", flush=True)
    concluido = subprocess.run(comando, cwd=str(ROOT), check=False,
                               env={**os.environ, **env})

    if concluido.returncode == 0:
        preservar_modelo_da_variante()
    return concluido.returncode


def preservar_modelo_da_variante() -> None:
    """Copia o T05 de quatro fontes para ``weights/`` com sufixo proprio.

    O diretorio da variante e area de trabalho e nao e versionado -- contem
    copias dos pesos de terceiros. Mas o modelo de fusao tem poucos KB e a
    Etapa 7 pede versionar os modelos treinados localmente, de modo que ele
    volta para ``weights/`` sob nome proprio, na mesma convencao ja usada por
    ``t05_fusion__variante_A.pkl``.

    ``weights/t05_fusion.pkl``, o de tres fontes, nao e tocado.
    """
    origem = PESOS_VARIANTE / "t05_fusion.pkl"
    if not origem.exists():
        print(f"[aviso] modelo de fusao nao encontrado em {origem}")
        return

    destino = WEIGHTS_DIR / "t05_fusion__quatro_fontes.pkl"
    shutil.copy2(origem, destino)
    print(f"\n[pesos] fusao de quatro fontes preservada em {destino}")
    print(f"[pesos] {WEIGHTS_DIR / 't05_fusion.pkl'} permanece o de tres fontes")


if __name__ == "__main__":
    raise SystemExit(main())
