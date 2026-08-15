"""Etapa 1 - Clonagem dos repositorios oficiais de T02 e T04.

T02 (SPAI) e T04 (geometria projetiva) sao executadas por meio do codigo e dos
modelos oficiais disponibilizados pelos autores. Cada repositorio exige um
ambiente proprio: o SPAI requer Python 3.11 e PyTorch com CUDA 12.4, versoes
distintas das adotadas neste projeto, razao pela qual a integracao ocorre por
subprocesso sobre ambientes virtuais isolados (mitigacao do risco R02).

Os pesos NAO sao baixados automaticamente: o checkpoint do SPAI e distribuido
via Google Drive e exige confirmacao manual. Os enderecos sao exibidos ao final.

Uso::

    python scripts/setup_external.py
    python scripts/setup_external.py --create-venvs
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import EXTERNAL, WEIGHTS_DIR, ensure_dirs   # noqa: E402

REPOSITORIES = {
    "spai": {
        "url": "https://github.com/mever-team/spai.git",
        "path": EXTERNAL.spai_repo,
        "python": "3.11",
        "weights_note": (
            "Baixe o checkpoint oficial indicado no README do repositorio "
            "(Google Drive) e grave em: {weights}/spai.pth"
        ),
    },
    "projective-geometry": {
        "url": "https://github.com/hanlinm2/projective-geometry.git",
        "path": EXTERNAL.geometry_repo,
        "python": "3.10",
        "weights_note": (
            "Baixe os pesos dos classificadores de perspective_fields, "
            "line_segment e object_shadow conforme o README e grave em: "
            "{weights}/projective_geometry/<componente>/"
        ),
    },
}


def clone(name: str, spec: dict) -> bool:
    path: Path = spec["path"]
    if path.exists():
        print(f"[{name}] ja presente em {path}")
        return True

    if shutil.which("git") is None:
        print(f"[{name}] ERRO: git nao encontrado no PATH")
        return False

    path.parent.mkdir(parents=True, exist_ok=True)
    print(f"[{name}] clonando {spec['url']} ...")
    completed = subprocess.run(
        ["git", "clone", "--depth", "1", spec["url"], str(path)],
        capture_output=True, text=True, check=False,
    )
    if completed.returncode != 0:
        print(f"[{name}] ERRO ao clonar:\n{completed.stderr.strip()[-500:]}")
        return False
    print(f"[{name}] clonado em {path}")
    return True


def create_venv(name: str, spec: dict) -> bool:
    """Cria um ambiente virtual isolado dentro do repositorio clonado.

    O adaptador de cada tecnica procura ``<repo>/.venv`` automaticamente, de
    modo que nenhuma configuracao adicional e necessaria apos esta etapa.
    """
    path: Path = spec["path"]
    if not path.exists():
        print(f"[{name}] repositorio ausente; clone antes de criar o ambiente")
        return False

    venv_path = path / ".venv"
    if venv_path.exists():
        print(f"[{name}] ambiente virtual ja existe em {venv_path}")
        return True

    print(f"[{name}] criando ambiente virtual (requer Python {spec['python']}) ...")
    completed = subprocess.run(
        [sys.executable, "-m", "venv", str(venv_path)],
        capture_output=True, text=True, check=False,
    )
    if completed.returncode != 0:
        print(f"[{name}] ERRO ao criar venv:\n{completed.stderr.strip()[-400:]}")
        return False

    requirements = path / "requirements.txt"
    if requirements.exists():
        pip = venv_path / ("Scripts/pip.exe" if sys.platform == "win32" else "bin/pip")
        print(f"[{name}] instalando dependencias de {requirements} ...")
        completed = subprocess.run(
            [str(pip), "install", "-r", str(requirements)],
            capture_output=True, text=True, check=False,
        )
        if completed.returncode != 0:
            print(f"[{name}] AVISO: falha na instalacao:\n{completed.stdout.strip()[-400:]}")
            print(f"[{name}] instale manualmente conforme o README do repositorio")
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepara os repositorios oficiais de T02 e T04")
    parser.add_argument("--create-venvs", action="store_true",
                        help="cria ambientes virtuais isolados dentro de cada repositorio")
    parser.add_argument("--only", choices=list(REPOSITORIES), default=None)
    args = parser.parse_args()

    ensure_dirs()
    WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)

    selection = [args.only] if args.only else list(REPOSITORIES)
    ok = True
    for name in selection:
        spec = REPOSITORIES[name]
        if not clone(name, spec):
            ok = False
            continue
        if args.create_venvs:
            create_venv(name, spec)

    print("\n" + "=" * 68)
    print("PESOS PRE-TREINADOS - download manual necessario")
    print("=" * 68)
    for name in selection:
        print(f"\n[{name}]")
        print("  " + REPOSITORIES[name]["weights_note"].format(weights=WEIGHTS_DIR))
    print("\nApos o download, verifique a disponibilidade com:")
    print("  python -c \"import sys; sys.path.insert(0,'.'); "
          "from src.techniques.t02_spai import T02SPAI; print(T02SPAI().is_available())\"")
    print("=" * 68)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
