"""Manifesto do corpus: indexacao, verificacao de integridade e particionamento.

Corresponde a Etapa 1 da Secao 3.5.2. O manifesto e um CSV unico com uma linha
por imagem, contendo caminho, rotulo, gerador de origem e particao. Centralizar
essa informacao evita que cada tecnica reimplemente a varredura do disco e
garante que todas avaliem exatamente as mesmas amostras.

Layout de diretorios esperado (dataset de Corvi et al., 2024)::

    data/corvi2024/
        real/<fonte>/*.png          fonte in {raise, fodb, imagenet, coco, open_images}
        fake/<gerador>/*.png        gerador in ALL_GENERATORS
"""

from __future__ import annotations

import csv
import hashlib
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
from PIL import Image

from ..configuracao import (
    ALL_GENERATORS,
    BENCHMARK_GENERATORS,
    DIFFUSION_GENERATORS,
    HELD_OUT_GENERATORS,
    IMAGE_SIZE,
    LABEL_FAKE,
    LABEL_REAL,
    SEED,
    FUSION_CALIBRATION_GENERATORS,
    FUSION_CALIBRATION_REAL_RATIO,
    SPLIT_RATIOS,
    TRAINING_GENERATORS,
)

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
MANIFEST_COLUMNS = ("path", "label", "generator", "source", "split", "width", "height")


@dataclass
class ManifestEntry:
    path: Path
    label: int
    generator: str      # "real" para imagens autenticas
    source: str         # base de origem (reais) ou familia do gerador
    split: str = ""
    width: int = 0
    height: int = 0


# ---------------------------------------------------------------------------
# Indexacao
# ---------------------------------------------------------------------------


def _family_of(generator: str) -> str:
    if generator in ("gigagan",):
        return "gan"
    if generator in ("midjourney_v5", "midjourney_v6_1"):
        return "commercial"
    return "diffusion"


def index_corpus(root: Path) -> list[ManifestEntry]:
    """Percorre o corpus e produz uma entrada por imagem encontrada."""
    root = Path(root)
    if not root.exists():
        raise FileNotFoundError(f"corpus nao encontrado em {root}")

    entries: list[ManifestEntry] = []

    real_root = root / "real"
    if real_root.exists():
        for source_dir in sorted(p for p in real_root.iterdir() if p.is_dir()):
            for path in sorted(source_dir.rglob("*")):
                if path.suffix.lower() in IMAGE_EXTENSIONS:
                    entries.append(ManifestEntry(path, LABEL_REAL, "real", source_dir.name))

    fake_root = root / "fake"
    if fake_root.exists():
        for generator_dir in sorted(p for p in fake_root.iterdir() if p.is_dir()):
            generator = generator_dir.name
            for path in sorted(generator_dir.rglob("*")):
                if path.suffix.lower() in IMAGE_EXTENSIONS:
                    entries.append(
                        ManifestEntry(path, LABEL_FAKE, generator, _family_of(generator))
                    )

    if not entries:
        raise FileNotFoundError(
            f"nenhuma imagem encontrada em {root}; verifique o layout real/ e fake/"
        )
    return entries


# ---------------------------------------------------------------------------
# Verificacao de integridade (Etapa 1)
# ---------------------------------------------------------------------------


def verify_integrity(
    entries: list[ManifestEntry],
    expected_size: int = IMAGE_SIZE,
    check_duplicates: bool = True,
) -> dict:
    """Verifica o corpus antes do inicio dos experimentos.

    Executa a contagem de amostras por classe e por gerador, a confirmacao de
    resolucao e formato dos arquivos, a deteccao de imagens corrompidas e a
    verificacao de duplicatas. Retorna um relatorio; nao levanta excecao, para
    que o usuario decida como tratar cada ocorrencia.
    """
    report: dict = {
        "n_total": len(entries),
        "by_label": Counter(),
        "by_generator": Counter(),
        "by_source": Counter(),
        "corrupted": [],
        "wrong_size": [],
        "duplicates": [],
        "missing_generators": [],
    }

    digests: dict[str, list[str]] = {}

    for entry in entries:
        report["by_label"][entry.label] += 1
        report["by_generator"][entry.generator] += 1
        report["by_source"][entry.source] += 1

        try:
            with Image.open(entry.path) as image:
                image.verify()          # detecta arquivos truncados/corrompidos
            with Image.open(entry.path) as image:
                entry.width, entry.height = image.size
                if check_duplicates:
                    # Hash do conteudo decodificado: detecta duplicatas mesmo
                    # com metadados ou nomes de arquivo distintos.
                    digest = hashlib.sha1(image.convert("RGB").tobytes()).hexdigest()
                    digests.setdefault(digest, []).append(str(entry.path))
        except Exception as error:                      # noqa: BLE001
            report["corrupted"].append({"path": str(entry.path), "error": str(error)})
            continue

        if expected_size and (entry.width, entry.height) != (expected_size, expected_size):
            report["wrong_size"].append(
                {"path": str(entry.path), "size": [entry.width, entry.height]}
            )

    if check_duplicates:
        report["duplicates"] = [paths for paths in digests.values() if len(paths) > 1]

    present = set(report["by_generator"]) - {"real"}
    report["missing_generators"] = sorted(set(ALL_GENERATORS) - present)
    report["unexpected_generators"] = sorted(present - set(ALL_GENERATORS))

    # Desbalanceamento entre geradores acima de 10% e o indicador do risco R07.
    counts = [report["by_generator"][g] for g in present] or [0]
    mean_count = float(np.mean(counts))
    report["generator_imbalance"] = (
        float(max(abs(c - mean_count) for c in counts) / mean_count) if mean_count else 0.0
    )
    report["balanced_classes"] = (
        report["by_label"][LABEL_REAL] > 0
        and abs(report["by_label"][LABEL_REAL] - report["by_label"][LABEL_FAKE])
        / max(report["by_label"][LABEL_REAL], 1) < 0.10
    )
    report["ok"] = (
        not report["corrupted"]
        and not report["missing_generators"]
        and report["generator_imbalance"] <= 0.10
    )
    return report


def format_report(report: dict) -> str:
    """Renderiza o relatorio de integridade para o console."""
    lines = [
        "=" * 68,
        "VERIFICACAO DE INTEGRIDADE DO CORPUS (Etapa 1)",
        "=" * 68,
        f"Total de imagens .............. {report['n_total']}",
        f"  reais ....................... {report['by_label'][LABEL_REAL]}",
        f"  sinteticas .................. {report['by_label'][LABEL_FAKE]}",
        f"Classes balanceadas ........... {'sim' if report['balanced_classes'] else 'NAO'}",
        f"Desbalanceamento entre gerad... {report['generator_imbalance']:.1%} (limite R07: 10%)",
        f"Arquivos corrompidos .......... {len(report['corrupted'])}",
        f"Resolucao inesperada .......... {len(report['wrong_size'])}",
        f"Grupos de duplicatas .......... {len(report['duplicates'])}",
    ]
    if report["missing_generators"]:
        lines.append(f"Geradores AUSENTES ............ {report['missing_generators']}")
    if report["unexpected_generators"]:
        lines.append(f"Geradores nao previstos ....... {report['unexpected_generators']}")

    lines.append("-" * 68)
    lines.append("Contagem por gerador:")
    for name, count in sorted(report["by_generator"].items()):
        lines.append(f"  {name:.<32} {count}")
    lines.append("=" * 68)
    lines.append("RESULTADO: " + ("APROVADO" if report["ok"] else "REQUER ATENCAO"))
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Particionamento
# ---------------------------------------------------------------------------


def assign_splits(
    entries: list[ManifestEntry],
    ratios: dict[str, float] = SPLIT_RATIOS,
    seed: int = SEED,
) -> list[ManifestEntry]:
    """Particiona em treinamento (70%), validacao (15%) e teste (15%).

    A estratificacao ocorre por (rotulo, gerador), garantindo que cada gerador
    esteja representado nas tres particoes na mesma proporcao, condicao
    necessaria para a avaliacao desagregada por gerador.
    """
    if abs(sum(ratios.values()) - 1.0) > 1e-9:
        raise ValueError(f"as proporcoes devem somar 1,0; obtido {sum(ratios.values())}")

    rng = np.random.default_rng(seed)
    strata: dict[tuple[int, str], list[ManifestEntry]] = {}
    for entry in entries:
        strata.setdefault((entry.label, entry.generator), []).append(entry)

    for group in strata.values():
        counts = _allocate(len(group), ratios)
        indices = rng.permutation(len(group))
        boundaries = (counts["train"], counts["train"] + counts["val"])
        for position, index in enumerate(indices):
            if position < boundaries[0]:
                group[index].split = "train"
            elif position < boundaries[1]:
                group[index].split = "val"
            else:
                group[index].split = "test"
    return entries


def _allocate(n: int, ratios: dict[str, float]) -> dict[str, int]:
    """Distribui ``n`` amostras entre as particoes pelo metodo do maior resto.

    O arredondamento independente de cada proporcao perde amostras em estratos
    pequenos: com n = 4, round(0,70*4) = 3 e round(0,15*4) = 1 esgotam o
    estrato e deixam o teste vazio. O metodo do maior resto preserva o total e,
    quando o estrato comporta, garante ao menos uma amostra em validacao e
    teste -- condicao necessaria para a avaliacao desagregada por gerador.
    """
    splits = ("train", "val", "test")
    exact = {split: ratios[split] * n for split in splits}
    counts = {split: int(np.floor(value)) for split, value in exact.items()}

    # Distribui as amostras restantes as particoes de maior parte fracionaria.
    remaining = n - sum(counts.values())
    by_remainder = sorted(splits, key=lambda s: exact[s] - counts[s], reverse=True)
    for position in range(remaining):
        counts[by_remainder[position % len(splits)]] += 1

    if n >= len(splits):
        for split in ("val", "test"):
            if counts[split] == 0:
                counts["train"] -= 1
                counts[split] = 1
    return counts


def ood_split(entries: list[ManifestEntry]) -> list[ManifestEntry]:
    """Reparticiona para o protocolo out-of-distribution (Secao 3.6.1).

    O treinamento usa exclusivamente o gerador do corpus de treino (difusao
    latente) e a avaliacao ocorre sobre os geradores do benchmark, nenhum deles
    visto no treinamento. As imagens reais mantem a particao original, de modo
    que nenhuma imagem real usada no treinamento reapareca no teste.

    Esta divisao e mais rigorosa que a descrita originalmente na Secao 3.6.1 --
    que separava apenas GigaGAN e Midjourney -- porque nenhum dos treze
    geradores avaliados participa do ajuste dos modelos.
    """
    for entry in entries:
        if entry.label == LABEL_REAL:
            continue
        if entry.generator in TRAINING_GENERATORS:
            # Remove o gerador de treino do conjunto de teste OOD.
            if entry.split == "test":
                entry.split = "train"
        elif entry.generator in BENCHMARK_GENERATORS:
            entry.split = "test"
        else:
            entry.split = "excluded"
    return entries


def ood_split_familias(entries: list[ManifestEntry], incluir_treino: bool = True) -> list[ManifestEntry]:
    """Protocolo OOD **conforme escrito** na Secao 3.6.1 do TCC 2.

    A Secao 3.6.1 define o protocolo de generalizacao como treinar nos dez
    geradores de difusao (Glide, Stable Diffusion 1.3, 1.4, 2, XL e 3, Flux,
    DALL-E 2, DALL-E 3 e Adobe Firefly) e avaliar nas familias mantidas de
    fora: GigaGAN, Midjourney v5 e Midjourney v6.1.

    Difere de :func:`ood_split`, que treina em um unico gerador e avalia nos
    treze restantes. As duas convivem de proposito: a comparacao entre elas
    separa o efeito da **diversidade** do conjunto de treinamento do efeito da
    mudanca de familia arquitetural. A variante desta funcao e mais fiel ao
    documento; a outra e mais rigorosa.

    ``incluir_treino`` mantem o gerador do corpus de treinamento
    (difusao latente) no ajuste. O TCC nomeia apenas os dez geradores do
    benchmark, mas difusao latente tambem e difusao, e exclui-lo reduziria o
    treinamento a 10.000 sinteticas contra as 30.000 da outra variante --
    tornando a comparacao entre protocolos indistinguivel de uma comparacao
    entre volumes de dados. Mantido por padrao, e a escolha esta declarada em
    documentacao/RESULTADOS.md.
    """
    treino_permitido = set(DIFFUSION_GENERATORS)
    if incluir_treino:
        treino_permitido |= set(TRAINING_GENERATORS)

    for entry in entries:
        if entry.label == LABEL_REAL:
            continue
        if entry.generator in HELD_OUT_GENERATORS:
            entry.split = "test"
        elif entry.generator in treino_permitido:
            # A avaliacao ocorre nas familias mantidas de fora; a porcao de
            # teste destes geradores seria desperdicada e vai para o treino.
            if entry.split == "test":
                entry.split = "train"
        else:
            entry.split = "excluded"
    return entries


def reserve_fusion_calibration(
    entries: list[ManifestEntry],
    generators: Sequence[str] = FUSION_CALIBRATION_GENERATORS,
    real_ratio: float = FUSION_CALIBRATION_REAL_RATIO,
    seed: int = SEED,
) -> list[ManifestEntry]:
    """Reserva um conjunto disjunto para calibrar o classificador de fusao.

    Aplicado APOS ``ood_split``. Move para a particao ``"fusion"`` todas as
    imagens sinteticas dos geradores indicados e uma fracao das imagens reais
    de teste, produzindo um terceiro conjunto que nao participa nem do
    treinamento das tecnicas-base nem da avaliacao final.

    Motivacao: ajustar T05 sobre escores de validacao in-distribution produz
    pesos calibrados para um regime em que todas as tecnicas-base acertam quase
    tudo, regime que nao se repete diante de geradores nao vistos. O conjunto
    de calibracao precisa exibir a mesma degradacao do conjunto de avaliacao.
    Ver documentacao/DECISOES_METODOLOGICAS.md.
    """
    generators = set(generators)
    rng = np.random.default_rng(seed)

    for entry in entries:
        if entry.label == LABEL_FAKE and entry.generator in generators:
            entry.split = "fusion"

    # Desvia parte das imagens reais de teste, preservando as duas classes no
    # conjunto de calibracao. Sem imagens reais, o classificador de fusao nao
    # teria como estimar o limiar entre as classes.
    real_test = [e for e in entries if e.label == LABEL_REAL and e.split == "test"]
    if real_test and real_ratio > 0:
        n_reserved = max(1, int(round(real_ratio * len(real_test))))
        for index in rng.choice(len(real_test), size=n_reserved, replace=False):
            real_test[index].split = "fusion"

    return entries


def filter_generators(
    entries: list[ManifestEntry],
    keep: Sequence[str],
) -> list[ManifestEntry]:
    """Mantem apenas as imagens reais e os geradores indicados.

    Usado no protocolo padrao (in-distribution), que avalia o desempenho sobre
    o mesmo gerador empregado no treinamento e portanto nao deve incluir os
    geradores do benchmark.
    """
    keep = set(keep)
    return [e for e in entries if e.label == LABEL_REAL or e.generator in keep]


# ---------------------------------------------------------------------------
# Persistencia
# ---------------------------------------------------------------------------


def write_manifest(entries: list[ManifestEntry], path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(MANIFEST_COLUMNS)
        for entry in entries:
            writer.writerow([
                str(entry.path), entry.label, entry.generator,
                entry.source, entry.split, entry.width, entry.height,
            ])
    return path


def read_manifest(path: Path, split: str | None = None) -> list[ManifestEntry]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"manifesto nao encontrado em {path}; execute automacao/prepare_dataset.py"
        )
    entries = []
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if split is not None and row["split"] != split:
                continue
            entries.append(ManifestEntry(
                path=Path(row["path"]),
                label=int(row["label"]),
                generator=row["generator"],
                source=row["source"],
                split=row["split"],
                width=int(row["width"] or 0),
                height=int(row["height"] or 0),
            ))
    return entries


def as_arrays(entries: list[ManifestEntry]) -> tuple[list[Path], np.ndarray, np.ndarray]:
    """Converte para ``(caminhos, rotulos, geradores)``."""
    paths = [entry.path for entry in entries]
    labels = np.asarray([entry.label for entry in entries], dtype=int)
    generators = np.asarray([entry.generator for entry in entries])
    return paths, labels, generators
