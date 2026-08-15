"""Encadeia o que depende de GPU, apos o termino do treino multi-semente.

Executa, em sequencia:

1. RNF01 para T02 (GPU e, opcionalmente, CPU)
2. Protocolo OOD por familias -- a variante escrita na Secao 3.6.1 do TCC 2:
   treino nos geradores de difusao, avaliacao em GigaGAN e Midjourney

Preservacao dos pesos
---------------------
A etapa 2 reajusta T01 e T03 sobre uma particao diferente e, sem cuidado,
sobrescreveria os modelos do protocolo padrao -- que sao os que a interface
carrega e os que produziram os resultados ja documentados.

A solucao usa ``TCC3_WEIGHTS_DIR``, previsto em ``src/config.py``: o
treinamento desta variante grava em um diretorio proprio. O checkpoint oficial
do SPAI e grande demais para copiar (891 MB), entao apenas os arquivos pequenos
sao replicados e o de T02 e referenciado por variavel de ambiente propria.

Uso::

    python scripts/pipeline_pos_multiseed.py --esperar-pid 12345
    python scripts/pipeline_pos_multiseed.py --sem-espera
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import DATA_DIR, RESULTS_DIR, WEIGHTS_DIR      # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
PYTHON = sys.executable
PESOS_OOD = ROOT / "weights_ood_familias"


def aguardar_log(log: Path, marcador: str, timeout_horas: float) -> bool:
    """Aguarda um marcador de conclusao aparecer no log de outra execucao.

    Espera por conteudo em vez de por PID: ``wmic`` foi removido das versoes
    recentes do Windows 11 e ``tasklist`` com filtro nao e confiavel quando o
    processo alvo foi iniciado por outro shell. O marcador so e escrito quando
    a etapa anterior conclui, o que torna a espera menos fragil.
    """
    limite = time.time() + timeout_horas * 3600
    print(f"[espera] aguardando '{marcador}' em {log.name} ...", flush=True)
    while time.time() < limite:
        if log.exists():
            try:
                if marcador in log.read_text(encoding="utf-8", errors="replace"):
                    print("[espera] marcador encontrado; prosseguindo", flush=True)
                    return True
            except OSError:
                pass
        time.sleep(120)
    print("[espera] tempo esgotado", flush=True)
    return False


def etapa(nome: str, comando: list[str], env: dict | None = None) -> bool:
    print("\n" + "=" * 74)
    print(f"ETAPA: {nome}")
    print("=" * 74, flush=True)
    print(" ".join(comando) + "\n", flush=True)

    inicio = time.perf_counter()
    concluido = subprocess.run(comando, cwd=str(ROOT), check=False,
                               env={**os.environ, **(env or {})})
    decorrido = (time.perf_counter() - inicio) / 60
    ok = concluido.returncode == 0
    print(f"\n>>> {nome}: {'ok' if ok else f'codigo {concluido.returncode}'} "
          f"em {decorrido:.1f} min", flush=True)
    return ok


def preparar_pesos_ood() -> dict[str, str]:
    """Diretorio de pesos separado, para nao sobrescrever o protocolo padrao."""
    PESOS_OOD.mkdir(parents=True, exist_ok=True)

    # O checkpoint do SPAI tem 891 MB; copia-lo seria desperdicio. T02 nao e
    # treinada, entao basta apontar o adaptador para o arquivo original.
    origem_spai = WEIGHTS_DIR / "spai.pth"
    destino_spai = PESOS_OOD / "spai.pth"
    if origem_spai.exists() and not destino_spai.exists():
        try:
            os.link(origem_spai, destino_spai)          # hard link: sem copia
            print(f"[pesos] link para {origem_spai.name}")
        except OSError:
            shutil.copy2(origem_spai, destino_spai)
            print(f"[pesos] copia de {origem_spai.name}")

    geometria = WEIGHTS_DIR / "projective_geometry"
    if geometria.exists() and not (PESOS_OOD / "projective_geometry").exists():
        shutil.copytree(geometria, PESOS_OOD / "projective_geometry")

    return {"TCC3_WEIGHTS_DIR": str(PESOS_OOD)}


def main() -> int:
    parser = argparse.ArgumentParser(description="Etapas de GPU apos o multi-semente")
    parser.add_argument("--esperar-log", type=Path,
                        default=ROOT / "logs" / "multiseed.log")
    parser.add_argument("--marcador", default="Metricas gravadas",
                        help="texto que sinaliza o fim da execucao anterior")
    parser.add_argument("--sem-espera", action="store_true")
    parser.add_argument("--timeout-horas", type=float, default=14.0)
    parser.add_argument("--incluir-t02-cpu", action="store_true")
    args = parser.parse_args()

    if not args.sem_espera:
        if not aguardar_log(args.esperar_log, args.marcador, args.timeout_horas):
            return 1

    resultados: dict[str, bool] = {}

    # --- 1. RNF01 para T02 ------------------------------------------------
    comando = [PYTHON, "-u", str(ROOT / "scripts" / "medir_rnf01.py"),
               "--limit", "1000", "--tecnicas", "T01", "T03", "T05", "T02"]
    if args.incluir_t02_cpu:
        comando.append("--incluir-t02-cpu")
    resultados["rnf01"] = etapa("RNF01 completo, com T02", comando)

    # --- 2. Protocolo OOD por familias -----------------------------------
    manifesto = DATA_DIR / "manifesto30k_ood_familias.csv"
    if not manifesto.exists():
        print(f"ERRO: manifesto ausente em {manifesto}")
        print("Execute antes: python scripts/prepare_dataset.py "
              "--corpus data/corvi2024_30k_norm --protocol ood_familias "
              f"--manifest {manifesto} --skip-duplicates")
        return 1

    env = preparar_pesos_ood()
    print(f"\n[pesos] treinamento desta variante grava em {PESOS_OOD}")
    print("[pesos] os modelos do protocolo padrao permanecem intactos\n")

    resultados["ood_familias"] = etapa(
        "OOD por familias (Secao 3.6.1) - treino e avaliacao",
        [PYTHON, "-u", str(ROOT / "scripts" / "run_experiments.py"),
         "--protocol", "ood", "--manifest", str(manifesto),
         "--fit", "--techniques", "T01", "T02", "T03", "T05"],
        env=env,
    )

    print("\n" + "=" * 74)
    print("RESUMO")
    print("=" * 74)
    for nome, ok in resultados.items():
        print(f"  {nome:<20} {'ok' if ok else 'FALHOU'}")
    print(f"\nResultados em {RESULTS_DIR}")
    return 0 if all(resultados.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
