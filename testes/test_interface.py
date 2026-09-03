"""Testes de interface (Secao 3.6.2).

Verificam os requisitos funcionais RF01 a RF06 e as regras de negocio RN01,
RN02, RN04, RN05 e RN07. Os casos incluem rejeicao de formatos invalidos
(TIFF, BMP, PDF renomeado como PNG), rejeicao de arquivos com tamanho superior
a 10 MB, verificacao de que nenhuma classificacao automatica e exibida (RN04),
verificacao da ordenacao das tecnicas (RN05) e verificacao de que a falha de um
modulo nao impede a exibicao dos resultados dos demais (RN07).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

pytest.importorskip("gradio", reason="os testes de interface exigem Gradio")

import interface.gradio_app as gradio_app                # noqa: E402
# TETO_T02 e TCC3_T02_NATIVA vivem no modulo de pre-processamento desde
# 03/09/2026. Remendar a interface nao teria efeito: quem le as constantes
# e envio_para_t02, e ela as le do proprio modulo.
import codigo.preprocessamento.envio_interface as envio                # noqa: E402
from interface.gradio_app import (      # noqa: E402
    DISPLAY_ORDER, MAX_FILE_BYTES, envio_normalizado, envio_para_t02,
    format_results, validate_upload,
)
from codigo.preprocessamento.normalizacao import resize_and_center_crop     # noqa: E402
from codigo.configuracao import IMAGE_SIZE                  # noqa: E402
from codigo.graficos import magnitude_spectrum_array     # noqa: E402
from codigo.tecnicas.base import TechniqueError     # noqa: E402


@pytest.fixture
def image_factory(tmp_path):
    def make(name: str, fmt: str, size: tuple[int, int] = (64, 64)) -> Path:
        rng = np.random.default_rng(42)
        array = rng.integers(0, 256, size=(*size[::-1], 3), dtype=np.uint8)
        path = tmp_path / name
        Image.fromarray(array).save(path, format=fmt)
        return path
    return make


# ---------------------------------------------------------------------------
# RN01 - formatos aceitos
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name,fmt", [
    ("imagem.png", "PNG"),
    ("imagem.jpg", "JPEG"),
    ("imagem.jpeg", "JPEG"),
    ("imagem.webp", "WEBP"),
])
def test_aceita_formatos_previstos(image_factory, name, fmt):
    valid, message = validate_upload(str(image_factory(name, fmt)))
    assert valid, message


@pytest.mark.parametrize("name,fmt", [("imagem.tiff", "TIFF"), ("imagem.bmp", "BMP")])
def test_rejeita_formatos_invalidos(image_factory, name, fmt):
    valid, message = validate_upload(str(image_factory(name, fmt)))
    assert not valid
    assert "RN01" in message


def test_rejeita_arquivo_renomeado(tmp_path):
    """PDF renomeado como PNG deve ser rejeitado pela inspecao do conteudo."""
    disguised = tmp_path / "documento.png"
    disguised.write_bytes(b"%PDF-1.4\n%fake pdf content\n")

    valid, message = validate_upload(str(disguised))
    assert not valid
    assert "corrompido" in message.lower() or "RN01" in message


def test_rejeita_bmp_renomeado_como_png(image_factory, tmp_path):
    source = image_factory("original.bmp", "BMP")
    disguised = tmp_path / "disfarcado.png"
    disguised.write_bytes(source.read_bytes())

    valid, message = validate_upload(str(disguised))
    assert not valid
    assert "BMP" in message


# ---------------------------------------------------------------------------
# RN02 - tamanho maximo
# ---------------------------------------------------------------------------


def test_rejeita_arquivo_acima_do_limite(tmp_path):
    oversized = tmp_path / "grande.png"
    oversized.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * (MAX_FILE_BYTES + 1))

    valid, message = validate_upload(str(oversized))
    assert not valid
    assert "RN02" in message and "10 MB" in message


def test_aceita_arquivo_no_limite(image_factory):
    path = image_factory("normal.png", "PNG")
    assert path.stat().st_size < MAX_FILE_BYTES
    assert validate_upload(str(path))[0]


# ---------------------------------------------------------------------------
# RF06 - validacao de entrada
# ---------------------------------------------------------------------------


def test_rejeita_entrada_vazia():
    assert not validate_upload(None)[0]
    assert not validate_upload("")[0]


def test_rejeita_arquivo_inexistente(tmp_path):
    valid, message = validate_upload(str(tmp_path / "nao_existe.png"))
    assert not valid
    assert "nao encontrado" in message.lower()


# ---------------------------------------------------------------------------
# RN04 e RN05 - apresentacao dos resultados
# ---------------------------------------------------------------------------


@pytest.fixture
def outcomes_completos():
    return {
        "T01": {"status": "ok", "score": 0.8312, "elapsed": 1.2},
        "T02": {"status": "ok", "score": 0.9145, "elapsed": 3.4},
        "T03": {"status": "ok", "score": 0.4021, "elapsed": 0.8},
        "T04": {"status": "ok", "score": 0.6633, "elapsed": 5.1},
        "T05": {"status": "ok", "score": 0.7788, "elapsed": 0.1, "n_sources": 4},
    }


def test_ordem_fixa_das_tecnicas(outcomes_completos):
    """RN05: as tecnicas sao exibidas em ordem fixa de identificacao."""
    rendered = format_results(outcomes_completos)
    positions = [rendered.index(f"**{t}**") for t in DISPLAY_ORDER]
    assert positions == sorted(positions)
    assert DISPLAY_ORDER == ("T01", "T02", "T03", "T04", "T05")


def test_exibe_escores_em_percentual(outcomes_completos):
    """RF03: escore de deteccao em percentual para cada tecnica."""
    rendered = format_results(outcomes_completos)
    assert "83.1%" in rendered
    assert "91.5%" in rendered
    assert "40.2%" in rendered


def test_sem_classificacao_binaria_automatica(outcomes_completos):
    """RN04: nenhuma classificacao binaria automatica e exibida."""
    rendered = format_results(outcomes_completos).lower()
    for termo in ("é sintética", "e sintetica", "imagem falsa", "veredito",
                  "classificação: real", "classificacao: real"):
        assert termo not in rendered
    assert "cabe ao usuario" in rendered or "cabe ao usuário" in rendered


def test_sumario_lista_nome_de_cada_tecnica(outcomes_completos):
    """RF04: sumario textual com nome e escore de cada tecnica."""
    rendered = format_results(outcomes_completos)
    assert "Benford" in rendered
    assert "SPAI" in rendered
    assert "Coocorrencia" in rendered or "Coocorrência" in rendered


# ---------------------------------------------------------------------------
# RN07 / RF08 - tolerancia a falhas
# ---------------------------------------------------------------------------


def test_falha_de_um_modulo_nao_oculta_os_demais(outcomes_completos):
    """RF08: exibe os resultados concluidos e sinaliza os indisponiveis."""
    outcomes_completos["T02"] = {"status": "erro", "detail": "CUDA out of memory"}
    rendered = format_results(outcomes_completos)

    assert "83.1%" in rendered          # T01 permanece visivel
    assert "40.2%" in rendered          # T03 permanece visivel
    assert "resultado não disponível" in rendered
    assert "CUDA out of memory" in rendered


def test_todos_os_modulos_indisponiveis():
    outcomes = {t: {"status": "indisponivel", "detail": "modelo ausente"} for t in DISPLAY_ORDER}
    rendered = format_results(outcomes)
    assert rendered.count("resultado não disponível") == 5


def test_modulo_ausente_do_dicionario_e_tratado():
    rendered = format_results({"T01": {"status": "ok", "score": 0.5, "elapsed": 0.1}})
    for technique_id in DISPLAY_ORDER:
        assert f"**{technique_id}**" in rendered


# ---------------------------------------------------------------------------
# Normalizacao do envio
# ---------------------------------------------------------------------------
#
# Ate 31/08/2026 a interface mandava o upload cru para as cinco tecnicas,
# enquanto todos os numeros do Capitulo 4 foram medidos sobre o corpus
# normalizado. Um envio de 1024 px invertia T01 e T02 na tela sobre a mesma
# imagem sintetica. Estes testes fixam a condicao de entrada.


@pytest.mark.parametrize("size", [(1024, 1024), (1024, 768), (300, 900), (64, 64)])
def test_envio_normalizado_entrega_256_em_rgb(image_factory, size):
    origem = image_factory("envio.png", "PNG", size=size)
    with envio_normalizado(origem) as caminho:
        with Image.open(caminho) as normalizada:
            assert normalizada.size == (IMAGE_SIZE, IMAGE_SIZE)
            assert normalizada.mode == "RGB"
            assert (normalizada.format or "").upper() == "PNG"


def test_envio_normalizado_reproduz_a_normalizacao_do_corpus(image_factory):
    """O pixel entregue as tecnicas e o mesmo que ``normalize_corpus`` gravaria."""
    origem = image_factory("envio.png", "PNG", size=(1024, 768))
    with Image.open(origem) as imagem:
        esperado = np.asarray(resize_and_center_crop(imagem, IMAGE_SIZE))

    with envio_normalizado(origem) as caminho:
        obtido = np.asarray(Image.open(caminho).convert("RGB"))

    assert np.array_equal(obtido, esperado)


def test_envio_ja_conforme_nao_copia(image_factory):
    """256 x 256 em RGB e o corpus inteiro: devolver o proprio arquivo mantem
    ``medir_rnf01_com_t04.py`` medindo o mesmo caminho de antes."""
    origem = image_factory("conforme.png", "PNG", size=(IMAGE_SIZE, IMAGE_SIZE))
    with envio_normalizado(origem) as caminho:
        assert caminho == origem


def test_envio_256_nao_rgb_e_convertido(tmp_path):
    """256 x 256 em escala de cinza nao esta conforme: falta o modo RGB."""
    origem = tmp_path / "cinza.png"
    Image.fromarray(np.full((IMAGE_SIZE, IMAGE_SIZE), 128, dtype=np.uint8)).save(origem)

    with envio_normalizado(origem) as caminho:
        assert caminho != origem
        with Image.open(caminho) as normalizada:
            assert normalizada.mode == "RGB"


def test_envio_normalizado_remove_o_temporario(image_factory):
    """RNF03: nenhum arquivo intermediario sobrevive a analise."""
    origem = image_factory("envio.png", "PNG", size=(800, 600))
    with envio_normalizado(origem) as caminho:
        assert caminho.exists()
        vazado = caminho
    assert not vazado.exists()
    assert not vazado.parent.exists()


def test_envio_normalizado_remove_o_temporario_em_excecao(image_factory):
    """A limpeza nao pode depender do sucesso da analise (RN07)."""
    origem = image_factory("envio.png", "PNG", size=(800, 600))
    vazado = None
    with pytest.raises(TechniqueError):
        with envio_normalizado(origem) as caminho:
            vazado = caminho
            raise TechniqueError("falha simulada de modulo")
    assert vazado is not None and not vazado.exists()


def test_envio_normalizado_fecha_o_descritor_da_origem(image_factory):
    """T02 abre a imagem num subprocesso; um descritor aberto do lado do Gradio
    impediria a leitura no Windows."""
    origem = image_factory("envio.png", "PNG", size=(1024, 1024))
    with envio_normalizado(origem):
        origem.unlink()          # so e possivel com o arquivo fechado
    assert not origem.exists()


def test_nota_de_normalizacao_aparece_no_sumario(outcomes_completos):
    """O leitor da tela precisa saber que nao ve o escore do arquivo cru."""
    rendered = format_results(outcomes_completos)
    assert f"{IMAGE_SIZE} × {IMAGE_SIZE}" in rendered
    assert "LANCZOS" in rendered


# ---------------------------------------------------------------------------
# Resolucao entregue a T02 - opcao (b) de 31/08/2026
# ---------------------------------------------------------------------------
#
# A T02 e espectral: normalizar para 256 px destroi a evidencia que ela explora.
# Em adobe_firefly_00002 (2688x1536, sintetica) ela mede 99,8% em resolucao
# nativa e 0,0% em 256 px. Mas a chamada nativa consumiu 5.931 MiB dos 6.144 da
# placa e derrubou os servicos T04 do WSL2 -- dai o teto.


def test_t02_recebe_o_original_quando_cabe_no_teto(image_factory):
    origem = image_factory("cabe.png", "PNG", size=(1024, 768))
    with envio_para_t02(origem) as caminho:
        assert caminho == origem          # sem reamostrar: sinal espectral intacto


def test_t02_reduz_ate_o_teto_preservando_proporcao(image_factory, monkeypatch):
    monkeypatch.setattr(envio, "TETO_T02", 1536)
    origem = image_factory("grande.png", "PNG", size=(2688, 1536))
    with envio_para_t02(origem) as caminho:
        assert caminho != origem
        with Image.open(caminho) as reduzida:
            assert max(reduzida.size) == 1536
            # 2688x1536 -> 1536x878; a proporcao original e preservada.
            assert reduzida.size == (1536, 878)
            assert reduzida.mode == "RGB"


def test_t02_respeita_teto_menor(image_factory, monkeypatch):
    monkeypatch.setattr(envio, "TETO_T02", 512)
    origem = image_factory("grande.png", "PNG", size=(2048, 1024))
    with envio_para_t02(origem) as caminho:
        with Image.open(caminho) as reduzida:
            assert reduzida.size == (512, 256)


def test_t02_desligada_por_variavel_de_ambiente(image_factory, monkeypatch):
    """TCC3_T02_NATIVA=0 devolve a interface ao comportamento de 31/08 pela manha."""
    monkeypatch.setattr(envio, "TCC3_T02_NATIVA", False)
    origem = image_factory("grande.png", "PNG", size=(2688, 1536))
    with envio_para_t02(origem) as caminho:
        assert caminho is None


def test_t02_remove_o_temporario(image_factory, monkeypatch):
    """RNF03, tambem no caminho da T02."""
    monkeypatch.setattr(envio, "TETO_T02", 512)
    origem = image_factory("grande.png", "PNG", size=(2048, 1024))
    with envio_para_t02(origem) as caminho:
        vazado = caminho
        assert vazado.exists()
    assert not vazado.exists()
    assert not vazado.parent.exists()


def test_t02_remove_o_temporario_em_excecao(image_factory, monkeypatch):
    monkeypatch.setattr(envio, "TETO_T02", 512)
    origem = image_factory("grande.png", "PNG", size=(2048, 1024))
    vazado = None
    with pytest.raises(TechniqueError):
        with envio_para_t02(origem) as caminho:
            vazado = caminho
            raise TechniqueError("falha simulada")
    assert vazado is not None and not vazado.exists()


def test_envio_ja_conforme_dispensa_a_segunda_passada(image_factory):
    """256 x 256 em RGB: os dois gerenciadores devolvem o mesmo arquivo, e
    ``run_analysis`` anula a segunda passada da T02 nesse caso."""
    origem = image_factory("conforme.png", "PNG", size=(IMAGE_SIZE, IMAGE_SIZE))
    with envio_normalizado(origem) as norm, envio_para_t02(origem) as para_t02:
        assert norm == para_t02 == origem


# --- o que a tela precisa dizer quando ha dois escores de T02 ---------------


@pytest.fixture
def outcomes_com_t02_nativa(outcomes_completos):
    outcomes_completos["T02"] = {
        "status": "ok",
        "score": 0.9977,            # resolucao nativa: vai para a tela
        "score_fusao": 0.0001,      # 256 px: alimenta a T05
        "elapsed": 43.2,
        "resolucao_nativa": True,
    }
    return outcomes_completos


def test_tela_marca_a_linha_da_t02_como_nativa(outcomes_com_t02_nativa):
    rendered = format_results(outcomes_com_t02_nativa)
    assert "99.8%" in rendered
    assert "resolução nativa" in rendered


def test_tela_declara_que_a_fusao_usa_o_escore_de_256(outcomes_com_t02_nativa):
    """Sem isto o leitor tentaria fechar a conta da T05 com o numero errado."""
    rendered = format_results(outcomes_com_t02_nativa)
    assert "0.0%" in rendered                    # o escore que alimenta a fusao
    assert "T05 continua sendo alimentada" in rendered


def test_sem_t02_nativa_a_tela_nao_menciona_resolucao(outcomes_completos):
    rendered = format_results(outcomes_completos)
    assert "resolução nativa" not in rendered
    assert "T05 continua sendo alimentada" not in rendered


def test_aviso_de_falha_da_passada_nativa_aparece(outcomes_completos):
    """RN07: se a resolucao nativa falhar por memoria, a analise segue inteira."""
    outcomes_completos["T02"] = {
        "status": "ok", "score": 0.0001, "elapsed": 40.0,
        "aviso": "resolucao nativa indisponivel (CUDA out of memory); "
                 "exibindo o escore de 256 px",
    }
    rendered = format_results(outcomes_completos)
    assert "CUDA out of memory" in rendered
    assert "resultado não disponível" not in rendered.split("T03")[0]


# ---------------------------------------------------------------------------
# RF05 - espectro de magnitude
# ---------------------------------------------------------------------------


def test_espectro_dimensoes_e_intervalo(image_factory):
    spectrum = magnitude_spectrum_array(image_factory("imagem.png", "PNG"))
    assert spectrum.shape == (256, 256)
    assert spectrum.dtype == np.uint8
    assert spectrum.max() <= 255


def test_espectro_reprodutivel(image_factory):
    """Consistencia entre a imagem submetida e o espectro exibido."""
    path = image_factory("imagem.png", "PNG")
    assert np.array_equal(magnitude_spectrum_array(path), magnitude_spectrum_array(path))


def test_espectro_simetria_hermitiana(tmp_path):
    """Para entrada real, o espectro de magnitude e simetrico em relacao ao centro."""
    rng = np.random.default_rng(0)
    array = rng.integers(0, 256, (256, 256, 3), dtype=np.uint8)
    path = tmp_path / "simetria.png"
    Image.fromarray(array).save(path)

    spectrum = magnitude_spectrum_array(path).astype(float)
    # Apos fftshift, o quadrante superior esquerdo espelha o inferior direito
    # (desconsiderando a linha e a coluna de Nyquist).
    core = spectrum[1:, 1:]
    assert np.allclose(core, core[::-1, ::-1], atol=1.5)


def test_espectro_detecta_padrao_periodico(tmp_path):
    """Um padrao periodico deve concentrar mais energia fora do centro."""
    grid_y, grid_x = np.meshgrid(np.arange(256), np.arange(256), indexing="ij")
    periodic = ((np.sin(np.pi * grid_y / 2) * np.sin(np.pi * grid_x / 2) + 1) * 127).astype(np.uint8)
    smooth = np.tile(np.linspace(0, 255, 256, dtype=np.uint8), (256, 1))

    paths = []
    for name, array in (("periodico.png", periodic), ("suave.png", smooth)):
        path = tmp_path / name
        Image.fromarray(np.stack([array] * 3, axis=2)).save(path)
        paths.append(path)

    def off_center_energy(path):
        spectrum = magnitude_spectrum_array(path).astype(float)
        mask = np.ones_like(spectrum, dtype=bool)
        mask[112:144, 112:144] = False        # exclui a regiao central
        return spectrum[mask].mean()

    assert off_center_energy(paths[0]) > off_center_energy(paths[1])
