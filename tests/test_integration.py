"""Testes de integracao e de interface (Secao 3.6.2).

Verificam o pipeline completo -- imagem de entrada, pre-processamento, extracao
de caracteristicas, classificacao e resultado -- para cada tecnica
individualmente, alem dos testes de falha controlada: cada modulo e desativado
sequencialmente para verificar que as demais tecnicas continuam produzindo
resultados corretamente (RN07).

Os testes sao executados em CPU para garantir reprodutibilidade independente de
hardware.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.manifest import (      # noqa: E402
    assign_splits, index_corpus, ood_split, verify_integrity,
    read_manifest, write_manifest,
)
from src.data.perturbations import PERTURBATIONS, build_perturbations, materialize  # noqa: E402
from src.techniques.base import TechniqueError                # noqa: E402
from src.techniques.t03_benford import T03Benford             # noqa: E402
from src.techniques.t05_fusion import T05Fusion, build_score_matrix   # noqa: E402


# ---------------------------------------------------------------------------
# Corpus de teste
# ---------------------------------------------------------------------------


@pytest.fixture
def tiny_corpus(tmp_path: Path) -> Path:
    """Corpus minimo com sinal detectavel, na estrutura real/ e fake/."""
    rng = np.random.default_rng(42)
    root = tmp_path / "corpus"

    for source in ("imagenet", "coco"):
        directory = root / "real" / source
        directory.mkdir(parents=True)
        for index in range(8):
            array = rng.integers(0, 256, size=(64, 64, 3), dtype=np.uint8)
            Image.fromarray(array).save(directory / f"{source}_{index}.png")

    for generator in ("glide", "gigagan"):
        directory = root / "fake" / generator
        directory.mkdir(parents=True)
        for index in range(8):
            # Bloco 2x2 replicado: artefato periodico de upsampling.
            half = rng.integers(0, 256, size=(32, 32, 3), dtype=np.uint8)
            array = np.repeat(np.repeat(half, 2, axis=0), 2, axis=1)
            Image.fromarray(array).save(directory / f"{generator}_{index}.png")

    return root


@pytest.fixture
def sample_paths(tiny_corpus: Path) -> tuple[list[Path], np.ndarray]:
    entries = index_corpus(tiny_corpus)
    return [e.path for e in entries], np.array([e.label for e in entries])


# ---------------------------------------------------------------------------
# Manifesto
# ---------------------------------------------------------------------------


def test_indexacao_encontra_todas_as_imagens(tiny_corpus):
    entries = index_corpus(tiny_corpus)
    assert len(entries) == 32
    assert sum(1 for e in entries if e.label == 0) == 16
    assert sum(1 for e in entries if e.label == 1) == 16


def test_indexacao_falha_em_diretorio_inexistente(tmp_path):
    with pytest.raises(FileNotFoundError):
        index_corpus(tmp_path / "inexistente")


def test_verificacao_de_integridade_detecta_corrompido(tiny_corpus):
    corrupted = tiny_corpus / "real" / "coco" / "corrompida.png"
    corrupted.write_bytes(b"isto nao e um PNG valido")

    report = verify_integrity(index_corpus(tiny_corpus), expected_size=0)
    assert len(report["corrupted"]) == 1
    assert "corrompida.png" in report["corrupted"][0]["path"]


def test_verificacao_detecta_resolucao_inesperada(tiny_corpus):
    report = verify_integrity(index_corpus(tiny_corpus), expected_size=256)
    # O corpus de teste usa 64x64, nao 256x256.
    assert len(report["wrong_size"]) == 32


def test_verificacao_detecta_duplicatas(tiny_corpus):
    import shutil

    original = next((tiny_corpus / "real" / "imagenet").glob("*.png"))
    shutil.copy(original, tiny_corpus / "real" / "coco" / "duplicata.png")

    report = verify_integrity(index_corpus(tiny_corpus), expected_size=0)
    assert len(report["duplicates"]) >= 1


def test_particionamento_estratificado_cobre_todos_geradores(tiny_corpus):
    entries = assign_splits(index_corpus(tiny_corpus))
    assert all(e.split in ("train", "val", "test") for e in entries)

    train = {e.generator for e in entries if e.split == "train"}
    assert "glide" in train and "gigagan" in train and "real" in train


def test_particionamento_reprodutivel(tiny_corpus):
    first = {str(e.path): e.split for e in assign_splits(index_corpus(tiny_corpus), seed=42)}
    second = {str(e.path): e.split for e in assign_splits(index_corpus(tiny_corpus), seed=42)}
    assert first == second


def test_particionamento_sem_sobreposicao(tiny_corpus):
    entries = assign_splits(index_corpus(tiny_corpus))
    splits: dict[str, set] = {"train": set(), "val": set(), "test": set()}
    for entry in entries:
        splits[entry.split].add(str(entry.path))
    assert not splits["train"] & splits["test"]
    assert not splits["train"] & splits["val"]
    assert not splits["val"] & splits["test"]


def test_particionamento_preserva_total(tiny_corpus):
    entries = assign_splits(index_corpus(tiny_corpus))
    assert len(entries) == 32
    counts = {"train": 0, "val": 0, "test": 0}
    for entry in entries:
        counts[entry.split] += 1
    assert sum(counts.values()) == 32


@pytest.mark.parametrize("group_size", [3, 4, 5, 7, 10, 13, 100])
def test_alocacao_nao_esvazia_particoes(group_size):
    """Estratos pequenos devem manter validacao e teste nao vazios.

    Regressao: o arredondamento independente de cada proporcao deixava o teste
    vazio para estratos de 4 amostras (round(0,7*4)=3 e round(0,15*4)=1).
    """
    from src.config import SPLIT_RATIOS
    from src.data.manifest import _allocate

    counts = _allocate(group_size, SPLIT_RATIOS)
    assert sum(counts.values()) == group_size
    assert all(value >= 0 for value in counts.values())
    assert counts["val"] >= 1 and counts["test"] >= 1
    assert counts["train"] >= 1


def test_alocacao_estrato_minimo():
    from src.config import SPLIT_RATIOS
    from src.data.manifest import _allocate

    # Com menos amostras que particoes nao ha garantia possivel.
    assert sum(_allocate(1, SPLIT_RATIOS).values()) == 1
    assert sum(_allocate(2, SPLIT_RATIOS).values()) == 2


def test_particionamento_gera_teste_com_todos_geradores(tiny_corpus):
    """Cada gerador deve estar representado no teste, para a analise desagregada."""
    entries = assign_splits(index_corpus(tiny_corpus))
    test_generators = {e.generator for e in entries if e.split == "test"}
    assert "glide" in test_generators
    assert "gigagan" in test_generators
    assert "real" in test_generators


def test_protocolo_ood_isola_geradores_do_benchmark(tiny_corpus):
    """O treino usa apenas o gerador do corpus de treino; o benchmark vai ao teste.

    O corpus minimo tem glide e gigagan, ambos do benchmark, e nenhuma imagem
    de latent_diffusion -- logo, nenhum gerador sintetico deve sobrar no treino.
    """
    entries = ood_split(assign_splits(index_corpus(tiny_corpus)))

    for generator in ("glide", "gigagan"):
        subset = [e for e in entries if e.generator == generator]
        assert subset, f"{generator} ausente do corpus de teste"
        assert all(e.split == "test" for e in subset), (
            f"{generator} pertence ao benchmark e deve ficar apenas no teste OOD"
        )

    sinteticas_no_treino = [
        e for e in entries if e.label == 1 and e.split in ("train", "val")
    ]
    assert not sinteticas_no_treino, (
        "nenhum gerador do benchmark pode participar do treinamento no protocolo OOD"
    )


def test_protocolo_ood_mantem_gerador_de_treino(tmp_path):
    """Imagens de latent_diffusion permanecem no treino e saem do teste."""
    rng = np.random.default_rng(7)
    root = tmp_path / "corpus"

    directory = root / "real" / "coco"
    directory.mkdir(parents=True)
    for index in range(10):
        Image.fromarray(rng.integers(0, 256, (32, 32, 3), dtype=np.uint8)).save(
            directory / f"coco_{index}.png"
        )

    for generator in ("latent_diffusion", "glide"):
        directory = root / "fake" / generator
        directory.mkdir(parents=True)
        for index in range(10):
            Image.fromarray(rng.integers(0, 256, (32, 32, 3), dtype=np.uint8)).save(
                directory / f"{generator}_{index}.png"
            )

    entries = ood_split(assign_splits(index_corpus(root)))

    latent = [e for e in entries if e.generator == "latent_diffusion"]
    assert latent and all(e.split in ("train", "val") for e in latent), (
        "o gerador de treino nao deve aparecer no teste OOD"
    )
    glide = [e for e in entries if e.generator == "glide"]
    assert glide and all(e.split == "test" for e in glide)


def test_calibracao_de_fusao_e_disjunta(tmp_path):
    """Os tres conjuntos -- treino, calibracao e avaliacao -- nao se sobrepoem.

    Cobre a decisao registrada em docs/DECISOES_METODOLOGICAS.md: T05 precisa de
    uma particao propria, com um gerador nao visto no treinamento e ausente da
    avaliacao final.
    """
    from src.data.manifest import reserve_fusion_calibration

    rng = np.random.default_rng(3)
    root = tmp_path / "corpus"

    directory = root / "real" / "coco"
    directory.mkdir(parents=True)
    for index in range(40):
        Image.fromarray(rng.integers(0, 256, (32, 32, 3), dtype=np.uint8)).save(
            directory / f"coco_{index:03d}.png"
        )
    for generator in ("latent_diffusion", "glide", "flux", "gigagan"):
        directory = root / "fake" / generator
        directory.mkdir(parents=True)
        for index in range(20):
            Image.fromarray(rng.integers(0, 256, (32, 32, 3), dtype=np.uint8)).save(
                directory / f"{generator}_{index:03d}.png"
            )

    entries = reserve_fusion_calibration(ood_split(assign_splits(index_corpus(root))))

    by_split: dict[str, set] = {}
    for entry in entries:
        by_split.setdefault(entry.split, set()).add(str(entry.path))

    assert "fusion" in by_split, "particao de calibracao nao foi criada"

    # Nenhuma imagem aparece em mais de uma particao.
    for a, b in (("train", "fusion"), ("fusion", "test"), ("train", "test")):
        assert not by_split.get(a, set()) & by_split.get(b, set()), (
            f"sobreposicao entre '{a}' e '{b}'"
        )

    # Todo o gerador de calibracao vai para a particao de calibracao.
    glide = [e for e in entries if e.generator == "glide"]
    assert glide and all(e.split == "fusion" for e in glide)

    # E some da avaliacao.
    assert not [e for e in entries if e.generator == "glide" and e.split == "test"]

    # A calibracao precisa das duas classes para estimar o limiar.
    rotulos = {e.label for e in entries if e.split == "fusion"}
    assert rotulos == {0, 1}, f"calibracao sem as duas classes: {rotulos}"

    # Os demais geradores do benchmark permanecem na avaliacao.
    for generator in ("flux", "gigagan"):
        assert [e for e in entries if e.generator == generator and e.split == "test"]


def test_calibracao_de_fusao_reprodutivel(tmp_path):
    from src.data.manifest import reserve_fusion_calibration

    rng = np.random.default_rng(5)
    root = tmp_path / "corpus"
    directory = root / "real" / "coco"
    directory.mkdir(parents=True)
    for index in range(30):
        Image.fromarray(rng.integers(0, 256, (32, 32, 3), dtype=np.uint8)).save(
            directory / f"coco_{index:03d}.png"
        )
    directory = root / "fake" / "glide"
    directory.mkdir(parents=True)
    for index in range(15):
        Image.fromarray(rng.integers(0, 256, (32, 32, 3), dtype=np.uint8)).save(
            directory / f"glide_{index:03d}.png"
        )

    def build():
        entries = reserve_fusion_calibration(ood_split(assign_splits(index_corpus(root))))
        return {str(e.path): e.split for e in entries}

    assert build() == build()


def test_manifesto_ida_e_volta(tiny_corpus, tmp_path):
    entries = assign_splits(index_corpus(tiny_corpus))
    path = write_manifest(entries, tmp_path / "manifesto.csv")

    recovered = read_manifest(path)
    assert len(recovered) == len(entries)
    assert {str(e.path) for e in recovered} == {str(e.path) for e in entries}

    train_only = read_manifest(path, split="train")
    assert all(e.split == "train" for e in train_only)


def test_manifesto_ausente_gera_erro_descritivo(tmp_path):
    with pytest.raises(FileNotFoundError, match="prepare_dataset"):
        read_manifest(tmp_path / "nao_existe.csv")


# ---------------------------------------------------------------------------
# Perturbacoes (protocolo de robustez)
# ---------------------------------------------------------------------------


def test_catalogo_de_perturbacoes_completo():
    names = {p.name for p in build_perturbations()}
    esperadas = {
        "clean",
        "jpeg_q50", "jpeg_q70", "jpeg_q85",
        "noise_sigma1", "noise_sigma3", "noise_sigma5",
        "resize_50", "resize_70", "resize_85",
    }
    assert names == esperadas


def test_perturbacoes_materializam_arquivos(sample_paths, tmp_path):
    paths, _ = sample_paths
    for name in ("clean", "jpeg_q50", "noise_sigma3", "resize_50"):
        written = materialize(paths[:4], PERTURBATIONS[name], tmp_path / name)
        assert len(written) == 4
        assert all(p.exists() and p.stat().st_size > 0 for p in written)


def test_redimensionamento_altera_resolucao(sample_paths, tmp_path):
    paths, _ = sample_paths
    written = materialize(paths[:2], PERTURBATIONS["resize_50"], tmp_path / "r50")
    with Image.open(written[0]) as image:
        assert image.size == (32, 32)      # metade de 64x64


def test_ruido_reprodutivel(sample_paths, tmp_path):
    """A mesma imagem deve receber o mesmo ruido em execucoes distintas (RNF02)."""
    paths, _ = sample_paths
    first = materialize(paths[:2], PERTURBATIONS["noise_sigma3"], tmp_path / "a")
    second = materialize(paths[:2], PERTURBATIONS["noise_sigma3"], tmp_path / "b")
    assert np.array_equal(np.asarray(Image.open(first[0])), np.asarray(Image.open(second[0])))


def test_nomes_unicos_evitam_colisao(tmp_path):
    """Imagens homonimas de geradores distintos nao devem se sobrescrever."""
    rng = np.random.default_rng(1)
    sources = []
    for index, generator in enumerate(("glide", "flux")):
        directory = tmp_path / generator
        directory.mkdir()
        path = directory / "imagem.png"
        Image.fromarray(rng.integers(0, 256, (32, 32, 3), dtype=np.uint8)).save(path)
        sources.append(path)

    written = materialize(sources, PERTURBATIONS["clean"], tmp_path / "saida")
    assert len(set(written)) == 2


# ---------------------------------------------------------------------------
# Pipeline de T03
# ---------------------------------------------------------------------------


def test_t03_pipeline_completo(sample_paths):
    paths, labels = sample_paths

    technique = T03Benford()
    features = technique.extract_features(paths)
    assert features.shape == (len(paths), 540)
    assert np.isfinite(features).all()

    technique.fit(paths, labels, features=features)
    probabilities = technique.predict_proba(paths, features=features)

    assert probabilities.shape == (len(paths),)
    assert ((probabilities >= 0) & (probabilities <= 1)).all()
    assert technique.best_params_["n_estimators"] in (100, 200, 500)


def test_t03_exige_ajuste_antes_da_predicao(sample_paths):
    paths, _ = sample_paths
    with pytest.raises(TechniqueError, match="fit"):
        T03Benford().predict_proba(paths[:2])


def test_t03_importancia_de_atributos_nomeada(sample_paths):
    paths, labels = sample_paths
    technique = T03Benford().fit(paths, labels)
    importances = technique.feature_importances()
    assert len(importances) == 540
    assert all(name.startswith("qf") for name in importances)


def test_t03_persistencia(sample_paths, tmp_path):
    paths, labels = sample_paths
    technique = T03Benford().fit(paths, labels)
    expected = technique.predict_proba(paths)

    technique.save(tmp_path / "t03.pkl")
    recovered = T03Benford().load(tmp_path / "t03.pkl")
    assert np.allclose(recovered.predict_proba(paths), expected)


# ---------------------------------------------------------------------------
# T05 - fusao e tolerancia a falhas
# ---------------------------------------------------------------------------


@pytest.fixture
def fusion_scores():
    rng = np.random.default_rng(42)
    y = np.array([0] * 60 + [1] * 60)
    scores = {
        # T01 e T02 informativas, T03 fraca, T04 aleatoria.
        "T01": np.clip(np.where(y == 1, rng.normal(0.75, 0.12, 120), rng.normal(0.25, 0.12, 120)), 0, 1),
        "T02": np.clip(np.where(y == 1, rng.normal(0.80, 0.10, 120), rng.normal(0.20, 0.10, 120)), 0, 1),
        "T03": np.clip(np.where(y == 1, rng.normal(0.58, 0.20, 120), rng.normal(0.42, 0.20, 120)), 0, 1),
        "T04": rng.uniform(0, 1, 120),
    }
    return scores, y


def test_matriz_de_escores_dimensoes(fusion_scores):
    scores, y = fusion_scores
    matrix = build_score_matrix(scores, len(y))
    assert matrix.shape == (120, 4)
    assert np.isfinite(matrix).all()


def test_matriz_de_escores_com_tecnica_ausente(fusion_scores):
    scores, y = fusion_scores
    del scores["T04"]
    matrix = build_score_matrix(scores, len(y))
    assert matrix.shape == (120, 4)
    assert np.isnan(matrix[:, 3]).all()          # coluna de T04
    assert np.isfinite(matrix[:, :3]).all()


def test_matriz_rejeita_forma_incorreta(fusion_scores):
    scores, y = fusion_scores
    scores["T01"] = scores["T01"][:10]
    with pytest.raises(TechniqueError, match="forma"):
        build_score_matrix(scores, len(y))


def test_t05_supera_tecnicas_isoladas(fusion_scores):
    """A fusao deve alcancar AUC ao menos equivalente a melhor tecnica isolada."""
    from sklearn.metrics import roc_auc_score

    scores, y = fusion_scores
    matrix = build_score_matrix(scores, len(y))

    fusion = T05Fusion().fit_scores(matrix, y)
    fused_auc = roc_auc_score(y, fusion.predict_proba_scores(matrix))
    best_individual = max(roc_auc_score(y, values) for values in scores.values())

    assert fused_auc >= best_individual - 0.02


def test_t05_opera_com_tecnica_indisponivel(fusion_scores):
    """RN07: a fusao deve funcionar mesmo sem os escores de um modulo."""
    scores, y = fusion_scores
    matrix = build_score_matrix(scores, len(y))
    fusion = T05Fusion().fit_scores(matrix, y)

    degraded = build_score_matrix({k: v for k, v in scores.items() if k != "T02"}, len(y))
    probabilities = fusion.predict_proba_scores(degraded)

    assert probabilities.shape == (120,)
    assert np.isfinite(probabilities).all()
    assert ((probabilities >= 0) & (probabilities <= 1)).all()


def test_t05_falha_controlada_de_cada_modulo(fusion_scores):
    """Desativa cada modulo sequencialmente e confirma que a fusao prossegue."""
    scores, y = fusion_scores
    fusion = T05Fusion().fit_scores(build_score_matrix(scores, len(y)), y)

    for disabled in ("T01", "T02", "T03", "T04"):
        remaining = {k: v for k, v in scores.items() if k != disabled}
        probabilities = fusion.predict_proba_scores(build_score_matrix(remaining, len(y)))
        assert np.isfinite(probabilities).all(), f"fusao falhou sem {disabled}"


def test_t05_pesos_por_dominio(fusion_scores):
    scores, y = fusion_scores
    fusion = T05Fusion().fit_scores(build_score_matrix(scores, len(y)), y)
    weights = fusion.contribution_weights()

    assert set(weights) == {"T01", "T02", "T03", "T04"}
    # T02 e a fonte mais informativa; T04 e ruido puro.
    assert weights["T02"] > weights["T04"]


def test_t05_persistencia(fusion_scores, tmp_path):
    scores, y = fusion_scores
    matrix = build_score_matrix(scores, len(y))
    fusion = T05Fusion().fit_scores(matrix, y)
    expected = fusion.predict_proba_scores(matrix)

    fusion.save(tmp_path / "t05.pkl")
    recovered = T05Fusion().load(tmp_path / "t05.pkl")
    assert np.allclose(recovered.predict_proba_scores(matrix), expected)


def test_t05_exige_ajuste_previo(fusion_scores):
    scores, y = fusion_scores
    with pytest.raises(TechniqueError):
        T05Fusion().predict_proba_scores(build_score_matrix(scores, len(y)))


# ---------------------------------------------------------------------------
# Indisponibilidade de T02 e T04
# ---------------------------------------------------------------------------


def test_t02_indisponivel_gera_erro_descritivo(tmp_path):
    from src.config import ExternalConfig
    from src.techniques.t02_spai import T02SPAI

    external = ExternalConfig(
        spai_repo=tmp_path / "ausente",
        spai_checkpoint=tmp_path / "ausente.pth",
    )
    technique = T02SPAI(external=external)

    available, reason = technique.is_available()
    assert not available
    assert "setup_external" in reason

    with pytest.raises(TechniqueError, match="indisponivel"):
        technique.predict_proba([tmp_path / "qualquer.png"])


def test_t04_indisponivel_gera_erro_descritivo(tmp_path):
    """Sem escores pre-extraidos, T04 continua integralmente indisponivel.

    ``precomputed`` aponta para um caminho inexistente de proposito: o padrao
    da classe e ``results/t04_escores_combined.csv``, que existe nesta maquina
    e tornaria a tecnica parcialmente disponivel.
    """
    from src.config import ExternalConfig
    from src.techniques.t04_geometry import T04ProjectiveGeometry

    external = ExternalConfig(geometry_repo=tmp_path / "ausente")
    technique = T04ProjectiveGeometry(
        external=external, precomputed=tmp_path / "sem_escores.csv")

    available, reason = technique.is_available()
    assert not available
    with pytest.raises(TechniqueError):
        technique.predict_proba([tmp_path / "qualquer.png"])


def test_t04_parcial_com_escores_pre_extraidos(tmp_path):
    """Com objeto-sombra extraido, T04 passa a contribuir com uma das tres.

    O escore vem do CSV; campos de perspectiva e segmentos de reta seguem sem
    extrator e entram como NaN, que ``nanmean`` ignora -- o mesmo mecanismo de
    degradacao previsto em RN07. O valor agregado, portanto, e o proprio escore
    de objeto-sombra.
    """
    from src.config import ExternalConfig
    from src.techniques.t04_geometry import T04ProjectiveGeometry

    escores = tmp_path / "escores.csv"
    escores.write_text(
        "arquivo,split,label,escore_t04\n"
        "coco_000001,test,0,0.12\n"
        "latent_diffusion_000001,test,1,0.93\n",
        encoding="utf-8",
    )

    technique = T04ProjectiveGeometry(
        external=ExternalConfig(geometry_repo=tmp_path / "ausente"),
        precomputed=escores,
    )

    available, reason = technique.is_available()
    assert available
    assert "parcial" in reason and "objeto-sombra" in reason

    caminhos = [tmp_path / "coco_000001.png", tmp_path / "latent_diffusion_000001.png"]
    probabilidades = technique.predict_proba(caminhos)
    assert probabilidades == pytest.approx([0.12, 0.93])

    # As duas representacoes sem extrator ficam explicitamente ausentes, e nao
    # imputadas: a analise de erros da Etapa 5 depende dessa distincao.
    componentes = technique.last_component_scores_
    assert np.isnan(componentes["perspective_fields"]).all()
    assert np.isnan(componentes["line_segment"]).all()
    assert componentes["object_shadow"] == pytest.approx([0.12, 0.93])


def test_t04_sem_variavel_de_ambiente_fica_indisponivel(tmp_path, monkeypatch):
    """O modo pre-extraido nao pode ligar sozinho.

    Regressao real: quando a disponibilidade era deduzida da existencia do CSV,
    T04 se anunciava disponivel na interface e falhava em toda imagem enviada --
    os escores cobrem o corpus deste trabalho, nunca um envio arbitrario. RN07
    exige o contrario: indisponibilidade declarada e as demais tecnicas seguindo.
    """
    import importlib

    escores = tmp_path / "escores.csv"
    escores.write_text("arquivo,split,label,escore_t04\ncoco_000001,test,0,0.12\n",
                       encoding="utf-8")

    monkeypatch.delenv("TCC3_T04_ESCORES", raising=False)
    modulo = importlib.reload(importlib.import_module("src.techniques.t04_geometry"))
    try:
        assert modulo.DEFAULT_PRECOMPUTED is None
        tecnica = modulo.T04ProjectiveGeometry()
        disponivel, motivo = tecnica.is_available()
        assert not disponivel
        assert "nao operam sobre pixels" in motivo
    finally:
        # Outros testes dependem do modulo no estado original.
        importlib.reload(modulo)


def test_t04_com_as_tres_representacoes(tmp_path):
    """Com os tres componentes extraidos, T04 agrega os tres.

    O agregador e a media; ausencia continua sendo ausencia, e nao zero -- uma
    coluna vazia some do vetor em vez de puxar a media para baixo.
    """
    from src.config import ExternalConfig
    from src.techniques.t04_geometry import T04ProjectiveGeometry

    escores = tmp_path / "componentes.csv"
    escores.write_text(
        "arquivo,split,label,object_shadow,perspective_fields,line_segment\n"
        "coco_000001,test,0,0.10,0.20,0.30\n"      # media 0,20
        "latent_diffusion_000001,test,1,0.90,,0.70\n",   # media 0,80, sem campos
        encoding="utf-8",
    )

    technique = T04ProjectiveGeometry(
        external=ExternalConfig(geometry_repo=tmp_path / "ausente"),
        precomputed=escores,
    )

    caminhos = [tmp_path / "coco_000001.png", tmp_path / "latent_diffusion_000001.png"]
    probabilidades = technique.predict_proba(caminhos)
    assert probabilidades == pytest.approx([0.20, 0.80])

    componentes = technique.last_component_scores_
    assert componentes["object_shadow"] == pytest.approx([0.10, 0.90])
    assert componentes["line_segment"] == pytest.approx([0.30, 0.70])
    # A segunda imagem nao tem campo de perspectiva: ausente, nao imputado.
    assert componentes["perspective_fields"][0] == pytest.approx(0.20)
    assert np.isnan(componentes["perspective_fields"][1])


def test_t04_recusa_imagem_sem_escore_extraido(tmp_path):
    """Imagem fora do subconjunto extraido nao deve receber escore inventado."""
    from src.config import ExternalConfig
    from src.techniques.t04_geometry import T04ProjectiveGeometry

    escores = tmp_path / "escores.csv"
    escores.write_text("arquivo,split,label,escore_t04\ncoco_000001,test,0,0.12\n",
                       encoding="utf-8")

    technique = T04ProjectiveGeometry(
        external=ExternalConfig(geometry_repo=tmp_path / "ausente"),
        precomputed=escores,
    )

    with pytest.raises(TechniqueError, match="todas as representacoes falharam"):
        technique.predict_proba([tmp_path / "imagem_nunca_extraida.png"])


def test_conversao_de_logits_para_probabilidade():
    from src.techniques.t02_spai import T02SPAI

    logits = np.array([-2.0, 0.0, 2.0])
    probabilities = T02SPAI._to_probabilities(logits)
    assert ((probabilities > 0) & (probabilities < 1)).all()
    assert probabilities[1] == pytest.approx(0.5)
    # A ordenacao e preservada, portanto a AUC nao se altera.
    assert np.array_equal(np.argsort(logits), np.argsort(probabilities))


def test_probabilidades_ja_normalizadas_sao_mantidas():
    from src.techniques.t02_spai import T02SPAI

    values = np.array([0.1, 0.5, 0.9])
    assert np.array_equal(T02SPAI._to_probabilities(values), values)
