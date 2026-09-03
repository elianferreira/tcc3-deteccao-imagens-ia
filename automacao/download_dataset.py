"""Etapa 1 - Aquisicao dos conjuntos de dados.

O corpus descrito na Secao 3.5.1 nao e distribuido como um pacote unico. Ele se
monta a partir de tres origens, com niveis distintos de automatizacao:

1. CONJUNTO DE TREINAMENTO -- Corvi et al. / DMimageDetection
   Arquivo unico de 20,45 GB em servidor HTTP direto. Baixavel por este script.
   Contem as imagens sinteticas de difusao latente e as LISTAS das imagens
   reais (COCO e LSUN), que precisam ser obtidas separadamente.

2. BENCHMARK DE AVALIACAO -- protocolo do SPAI (13 geradores)
   Composto por seis fontes. Tres sao baixaveis diretamente; tres exigem conta
   ou formulario de cadastro e precisam de acao manual.

3. GERADORES ADICIONAIS (Stable Diffusion 3, Midjourney v6.1, GigaGAN, Flux)
   Distribuidos via Google Drive, que exige token de confirmacao para arquivos
   grandes. Requer ``gdown`` ou download manual pelo navegador.

Uso::

    python automacao/download_dataset.py --list          # o que falta e de onde vem
    python automacao/download_dataset.py --auto          # baixa tudo que e automatizavel
    python automacao/download_dataset.py --only synthbuster
    python automacao/download_dataset.py --verify        # confere integridade do que ja existe
"""

from __future__ import annotations

import argparse
import hashlib
import shutil
import subprocess
import sys
import zipfile
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from codigo.configuracao import DATA_DIR, ensure_dirs      # noqa: E402

DOWNLOAD_DIR = DATA_DIR / "downloads"


@dataclass
class Source:
    key: str
    description: str
    url: str
    filename: str = ""
    #: True quando um GET simples basta; False exige conta ou formulario.
    automatic: bool = True
    manual_note: str = ""
    sha256: str = ""
    size_gb: float = 0.0
    provides: list[str] = field(default_factory=list)
    #: ID do Google Drive. Arquivos grandes exigem token de confirmacao, que o
    #: gdown negocia automaticamente; curl sozinho recebe a pagina de aviso.
    gdrive_id: str = ""


SOURCES: list[Source] = [
    # ------------------------------------------------------------------
    # 1. Conjunto de treinamento
    # ------------------------------------------------------------------
    Source(
        key="latent_diffusion_train",
        description="Treinamento: 200k sinteticas (difusao latente) + listas das reais",
        url="https://www.grip.unina.it/download/prog/DMimageDetection/latent_diffusion_trainingset.zip",
        filename="latent_diffusion_trainingset.zip",
        automatic=True,
        size_gb=20.45,
        provides=["fake/latent_diffusion", "listas de COCO e LSUN"],
    ),
    # ------------------------------------------------------------------
    # 2. Benchmark de avaliacao
    # ------------------------------------------------------------------
    Source(
        # A API do Zenodo expoe o conteudo por URL direta, sem cadastro: o
        # registro 10066460 contem um unico arquivo.
        key="synthbuster",
        description="Synthbuster: sinteticas de multiplos geradores comerciais",
        url="https://zenodo.org/api/records/10066460/files/synthbuster.zip/content",
        filename="synthbuster.zip",
        automatic=True,
        # 12.372.557.226 bytes, conforme o cabecalho Content-Range do servidor.
        size_gb=12_372_557_226 / 1024**3,
        provides=["fake/dalle2", "fake/dalle3", "fake/adobe_firefly",
                  "fake/midjourney_v5", "fake/glide", "fake/stable_diffusion_*"],
    ),
    Source(
        key="coco2017",
        description="COCO 2017 (imagens reais)",
        url="http://images.cocodataset.org/zips/val2017.zip",
        filename="coco_val2017.zip",
        automatic=True,
        # 815.585.330 bytes conforme o servidor.
        size_gb=815_585_330 / 1024**3,
        provides=["real/coco"],
    ),
    Source(
        key="open_images",
        description="Open Images V7 (imagens reais)",
        url="https://storage.googleapis.com/openimages/web/download_v7.html",
        automatic=False,
        manual_note=(
            "Download por subconjunto, via a ferramenta oficial. Para o TCC "
            "bastam ~1.000 imagens do conjunto de teste."
        ),
        provides=["real/open_images"],
    ),
    Source(
        key="fodb",
        description="FODB: Forchheim Image Database (imagens reais de camera)",
        url="https://faui1-files.cs.fau.de/public/mmsec/datasets/fodb/",
        automatic=False,
        manual_note="Indice de diretorio HTTP. Escolha os arquivos e baixe diretamente.",
        provides=["real/fodb"],
    ),
    Source(
        key="imagenet",
        description="ImageNet (imagens reais)",
        url="https://www.kaggle.com/c/imagenet-object-localization-challenge/data",
        automatic=False,
        manual_note=(
            "EXIGE CONTA KAGGLE e aceite das regras da competicao. Com a API "
            "configurada:  kaggle competitions download -c "
            "imagenet-object-localization-challenge"
        ),
        provides=["real/imagenet"],
    ),
    Source(
        key="raise",
        description="RAISE-1k (imagens reais RAW)",
        url="https://loki.disi.unitn.it/RAISE/download.html",
        automatic=False,
        manual_note=(
            "EXIGE FORMULARIO DE CADASTRO. Preencha no site e use o link "
            "enviado por e-mail."
        ),
        provides=["real/raise"],
    ),
    # ------------------------------------------------------------------
    # 3. Geradores adicionais
    # ------------------------------------------------------------------
    Source(
        key="extra_generators",
        description="Stable Diffusion 3, Midjourney v6.1, GigaGAN e Flux",
        url="https://drive.google.com/file/d/1no5T89h97TZvAKNCHKt2PQfKZ1UDtbI4/view?usp=sharing",
        filename="extra_generators.zip",
        gdrive_id="1no5T89h97TZvAKNCHKt2PQfKZ1UDtbI4",
        automatic=True,
        size_gb=3.72,
        provides=["fake/stable_diffusion_3", "fake/midjourney_v6_1",
                  "fake/gigagan", "fake/flux"],
    ),
    Source(
        key="dm_testset",
        description="Conjunto de teste sintetico do DMimageDetection",
        url="https://drive.google.com/file/d/1grvgKiIq0ny8ImQzSUXPk3nd-AMEDjNb/view?usp=share_link",
        filename="dm_testset.zip",
        gdrive_id="1grvgKiIq0ny8ImQzSUXPk3nd-AMEDjNb",
        automatic=True,
        provides=["fake/glide", "fake/stable_diffusion_1_3", "fake/stable_diffusion_1_4"],
    ),
]

BY_KEY = {source.key: source for source in SOURCES}


# ---------------------------------------------------------------------------
# Download
# ---------------------------------------------------------------------------


def download(source: Source, destination_dir: Path, force: bool = False) -> Path | None:
    """Baixa uma fonte automatizavel, com retomada em caso de queda."""
    if not source.automatic:
        print(f"[{source.key}] requer acao manual; use --list para as instrucoes")
        return None

    destination_dir.mkdir(parents=True, exist_ok=True)
    target = destination_dir / (source.filename or source.url.rsplit("/", 1)[-1])

    if target.exists() and not force:
        # A verificacao estrutural decide se o arquivo esta completo; o tamanho
        # declarado e estimativa e produziria falso negativo (ou falso positivo)
        # em arquivo integro.
        if verify(source, destination_dir):
            return target
        size_gb = target.stat().st_size / 1024**3
        print(f"[{source.key}] incompleto ({size_gb:.2f} GB), retomando do ponto de parada")

    print(f"[{source.key}] baixando {source.size_gb or '?'} GB de {source.url}")

    if source.gdrive_id:
        # O Google Drive intercepta arquivos grandes com uma pagina de aviso
        # antivirus; curl salvaria esse HTML no lugar do conteudo. O gdown
        # negocia o token de confirmacao e retoma downloads parciais.
        command = [
            sys.executable, "-m", "gdown", "--continue",
            "-O", str(target), source.gdrive_id,
        ]
    else:
        if shutil.which("curl") is None:
            print(f"[{source.key}] ERRO: curl nao encontrado no PATH")
            return None
        # -C - retoma de onde parou; --retry-all-errors cobre quedas de rede.
        command = [
            "curl", "-L", "-C", "-", "--retry", "5", "--retry-delay", "10",
            "--retry-all-errors", "-o", str(target), source.url,
        ]

    completed = subprocess.run(command, check=False)
    if completed.returncode != 0:
        print(f"[{source.key}] FALHOU (codigo {completed.returncode}); reexecute para retomar")
        return None

    print(f"[{source.key}] concluido: {target} ({target.stat().st_size / 1024**3:.2f} GB)")
    return target


def sha256_of(path: Path, chunk_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify(source: Source, destination_dir: Path) -> bool:
    """Confere a integridade do arquivo baixado.

    A ordem das verificacoes importa. ``size_gb`` e uma estimativa retirada da
    documentacao das fontes e nao coincide exatamente com o tamanho real do
    arquivo; usa-la como criterio de aprovacao produz falso negativo em
    download integro. Por isso o teste estrutural -- SHA256 quando conhecido,
    integridade do ZIP quando aplicavel -- e a autoridade, e o tamanho serve
    apenas para diagnosticar um arquivo visivelmente truncado.
    """
    target = destination_dir / (source.filename or source.url.rsplit("/", 1)[-1])
    if not target.exists():
        print(f"[{source.key}] ausente")
        return False

    size_gb = target.stat().st_size / 1024**3

    if source.sha256:
        actual = sha256_of(target)
        if actual != source.sha256:
            print(f"[{source.key}] SHA256 DIVERGENTE\n  esperado: {source.sha256}\n  obtido:   {actual}")
            return False
        print(f"[{source.key}] ok ({size_gb:.2f} GB, SHA256 confere)")
        return True

    if target.suffix == ".zip":
        # Verificacao autoritativa: um ZIP truncado ou corrompido falha aqui,
        # mesmo que o tamanho pareca plausivel.
        try:
            with zipfile.ZipFile(target) as archive:
                bad = archive.testzip()
                n_entries = len(archive.namelist())
            if bad is not None:
                print(f"[{source.key}] ZIP CORROMPIDO na entrada {bad}")
                return False
            print(f"[{source.key}] ok ({size_gb:.2f} GB, ZIP integro, {n_entries} entradas)")
            return True
        except zipfile.BadZipFile:
            print(f"[{source.key}] ZIP INVALIDO ou incompleto ({size_gb:.2f} GB baixados)")
            return False

    # Sem SHA256 nem estrutura verificavel, resta comparar com a estimativa.
    if source.size_gb and size_gb < source.size_gb * 0.95:
        print(f"[{source.key}] possivelmente incompleto: {size_gb:.2f} de ~{source.size_gb:.2f} GB")
        return False

    print(f"[{source.key}] presente ({size_gb:.2f} GB, sem verificacao estrutural disponivel)")
    return True


def extract(source: Source, destination_dir: Path, output_dir: Path) -> None:
    target = destination_dir / (source.filename or source.url.rsplit("/", 1)[-1])
    if not target.exists() or target.suffix != ".zip":
        return
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"[{source.key}] extraindo para {output_dir} ...")
    with zipfile.ZipFile(target) as archive:
        archive.extractall(output_dir)
    print(f"[{source.key}] extraido")


# ---------------------------------------------------------------------------
# Relatorio
# ---------------------------------------------------------------------------


def print_plan() -> None:
    automatic = [s for s in SOURCES if s.automatic]
    manual = [s for s in SOURCES if not s.automatic]

    print("=" * 74)
    print("AQUISICAO DO CORPUS - Etapa 1")
    print("=" * 74)

    print("\nAUTOMATICO (este script baixa):\n")
    for source in automatic:
        status = "presente" if (DOWNLOAD_DIR / source.filename).exists() else "faltando"
        print(f"  [{status:>8}] {source.key:<24} {source.size_gb:>6.2f} GB  {source.description}")

    print("\nMANUAL (exige conta, formulario ou token do Google Drive):\n")
    for source in manual:
        print(f"  {source.key:<24} {source.description}")
        print(f"      URL:  {source.url}")
        print(f"      Nota: {source.manual_note}")
        print(f"      Supre: {', '.join(source.provides)}")
        print()

    print("=" * 74)
    print("Apos reunir tudo, organize em data/corvi2024/ no layout real/ e fake/")
    print("descrito no README e execute:")
    print("  python automacao/prepare_dataset.py --corpus data/corvi2024")
    print("=" * 74)


def main() -> int:
    parser = argparse.ArgumentParser(description="Baixa os conjuntos de dados do TCC")
    parser.add_argument("--list", action="store_true", help="mostra o plano de aquisicao")
    parser.add_argument("--auto", action="store_true", help="baixa tudo que e automatizavel")
    parser.add_argument("--only", choices=list(BY_KEY), default=None)
    parser.add_argument("--verify", action="store_true", help="confere o que ja foi baixado")
    parser.add_argument("--extract", action="store_true", help="extrai os ZIPs baixados")
    parser.add_argument("--force", action="store_true", help="rebaixa mesmo se ja existir")
    parser.add_argument("--output", type=Path, default=DATA_DIR / "corvi2024")
    args = parser.parse_args()

    ensure_dirs()
    DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)

    if args.list or not (args.auto or args.only or args.verify or args.extract):
        print_plan()
        return 0

    selection = [BY_KEY[args.only]] if args.only else SOURCES

    if args.verify:
        results = [verify(s, DOWNLOAD_DIR) for s in selection if s.automatic]
        return 0 if all(results) else 1

    if args.auto or args.only:
        for source in selection:
            download(source, DOWNLOAD_DIR, force=args.force)

    if args.extract:
        for source in selection:
            if source.automatic:
                extract(source, DOWNLOAD_DIR, args.output)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
