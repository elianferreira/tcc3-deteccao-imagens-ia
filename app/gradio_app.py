"""Interface grafica demonstrativa (Etapa 6 da Secao 3.5.2).

A interface recebe uma imagem nos formatos PNG, JPEG ou WebP e executa os
modulos de cada tecnica de forma independente, exibindo para cada uma a
probabilidade estimada em percentual e o espectro de magnitude correspondente.

Regras de negocio atendidas:

RN01  formatos aceitos: PNG, JPEG, WebP
RN02  tamanho maximo de 10 MB
RN04  exibe apenas probabilidades, sem classificacao binaria automatica
RN05  tecnicas apresentadas em ordem fixa de identificacao
RN07  a falha de um modulo nao impede a exibicao dos resultados dos demais

A interface tem carater demonstrativo-experimental e nao se destina a
implantacao em ambiente de producao.
"""

from __future__ import annotations

import contextlib
import os
import sys
import tempfile
import time
from collections.abc import Iterator
from pathlib import Path

import gradio as gr
import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.normalize_corpus import resize_and_center_crop      # noqa: E402
from src.config import (                                         # noqa: E402
    EXTERNAL, IMAGE_SIZE, TECHNIQUE_NAMES, WEIGHTS_DIR,
)
from src.plots import magnitude_spectrum_array                  # noqa: E402
from src.techniques.base import TechniqueError                  # noqa: E402
from src.techniques.t02_spai import T02SPAI                     # noqa: E402
from src.techniques.t03_benford import T03Benford               # noqa: E402
from src.techniques.t04_geometry import T04ProjectiveGeometry   # noqa: E402
from src.techniques.t05_fusion import T05Fusion, build_score_matrix   # noqa: E402

# RN01 e RN02
ALLOWED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp"}
ALLOWED_FORMATS = {"PNG", "JPEG", "WEBP"}
MAX_FILE_BYTES = 10 * 1024 * 1024

DISPLAY_ORDER = ("T01", "T02", "T03", "T04", "T05")   # RN05

# ---------------------------------------------------------------------------
# Politica de resolucao por tecnica -- opcao (b) de 31/08/2026
# ---------------------------------------------------------------------------
#
# Cada tecnica recebe a imagem na condicao em que foi medida. T01, T03 e T04
# foram treinadas e avaliadas em 256x256; a T02 e espectral e foi verificada em
# resolucao nativa (secao 4.9: o controle do glide da 0,9626 contra 0,902
# publicado).
#
# Medido em adobe_firefly_00002 (2688x1536, sintetica):
#
#     tecnica    nativa    256 px
#     T01          2,5%     97,7%
#     T02         99,8%      0,0%
#     T04         69,8%     69,7%   (mesma resposta, e 12x mais cara em nativa)
#
# Nenhuma resolucao unica funciona: com tudo normalizado a T05 da 2,5%, com tudo
# nativo da 19,5% -- as duas erram. Com a T02 em nativa e o resto normalizado, a
# T05 da 99,3%.
#
# A T05 continua sendo alimentada pelo escore de 256 px, e nao pelo nativo. A
# fusao foi ajustada nessa distribuicao, e troca-la aqui mediria o reajuste em
# vez da mudanca (armadilha 6 de ESTADO_ATUAL.md). Por isso a T02 e avaliada
# duas vezes: o nativo vai para a tela, o de 256 px vai para a fusao.
#
# TETO: a T02 em 4,13 MPx consumiu 5.931 MiB dos 6.144 da placa e derrubou os
# servicos T04 do WSL2 por esgotamento de memoria. A RN02 aceita 10 MB, que
# podem ser bem maiores. Sem teto isto nao e operavel neste hardware.
TCC3_T02_NATIVA = os.environ.get("TCC3_T02_NATIVA", "1") != "0"
TETO_T02 = int(os.environ.get("TCC3_T02_TETO", "1536"))


class DetectionService:
    """Carrega os modelos uma unica vez e executa os modulos isoladamente.

    Os modelos sao carregados em memoria na inicializacao da interface; erros
    de inferencia em um modulo especifico sao capturados e reportados ao
    usuario sem interromper a execucao dos demais modulos.
    """

    def __init__(self) -> None:
        self.techniques: dict[str, object] = {}
        self.load_errors: dict[str, str] = {}
        # Qual modelo de fusao foi carregado; depende de T04 estar disponivel.
        self.fusion_model_name: str = ""
        self._load_all()
        self._aquecer()

    def _aquecer(self) -> None:
        """Uma inferencia descartavel, para que a primeira imagem real nao pague
        o arranque a frio.

        Medido em 30/08/2026: sem aquecimento, a primeira analise custou 40,5 s
        -- **acima do teto de 30 s do RNF01** --, contra 20,7 s em regime. O
        excedente e a criacao do contexto CUDA e a escolha de algoritmos da
        cuDNN, que acontecem uma vez e ficam em cache no processo.

        Nao e cosmetico: sem isto, o primeiro usuario a enviar uma imagem depois
        de a interface subir veria o requisito violado.

        Falhas aqui sao silenciosas de proposito. O aquecimento e otimizacao; se
        ele nao funcionar, a analise real ainda funciona -- so mais devagar na
        primeira vez. Derrubar a interface por causa disso seria pior.
        """
        import tempfile

        import numpy as np
        from PIL import Image

        try:
            with tempfile.TemporaryDirectory(prefix="aquecimento_") as pasta:
                alvo = Path(pasta) / "aquecimento.png"
                gerador = np.random.default_rng(0)
                Image.fromarray(
                    gerador.integers(0, 256, size=(256, 256, 3), dtype=np.uint8)
                ).save(alvo)
                self.analyze(alvo)
        except Exception:                                # noqa: BLE001, S110
            pass

    def _load_all(self) -> None:
        # T01 - checkpoint treinado localmente
        try:
            from src.techniques.t01_cooccurrence import T01Cooccurrence

            technique = T01Cooccurrence()
            checkpoint = WEIGHTS_DIR / "t01_cooccurrence.pt"
            if checkpoint.exists():
                self.techniques["T01"] = technique.load(checkpoint)
            else:
                self.load_errors["T01"] = f"checkpoint ausente em {checkpoint}"
        except Exception as error:                      # noqa: BLE001
            self.load_errors["T01"] = str(error)

        # T02 - pipeline oficial do SPAI
        try:
            spai = T02SPAI()
            available, reason = spai.is_available()
            if available:
                self.techniques["T02"] = spai
            else:
                self.load_errors["T02"] = reason
        except Exception as error:                      # noqa: BLE001
            self.load_errors["T02"] = str(error)

        # T03 - Random Forest treinado localmente
        try:
            model_path = WEIGHTS_DIR / "t03_benford.pkl"
            if model_path.exists():
                self.techniques["T03"] = T03Benford().load(model_path)
            else:
                self.load_errors["T03"] = f"modelo ausente em {model_path}"
        except Exception as error:                      # noqa: BLE001
            self.load_errors["T03"] = str(error)

        # T04 - pipeline oficial de geometria projetiva
        try:
            geometry = T04ProjectiveGeometry()
            available, reason = geometry.is_available()
            if available:
                self.techniques["T04"] = geometry
            else:
                self.load_errors["T04"] = reason
        except Exception as error:                      # noqa: BLE001
            self.load_errors["T04"] = str(error)

        # T05 - classificador de fusao
        #
        # O modelo precisa casar com as fontes que de fato produzem escore.
        # Ha dois ajustados:
        #
        #   t05_fusion.pkl                 tres fontes; o coeficiente de T04 e
        #                                  exatamente 0,0, porque T04 nunca
        #                                  esteve presente no ajuste
        #   t05_fusion__quatro_fontes.pkl  quatro fontes; peso de T04 = -0,2512
        #
        # Carregar o de tres fontes com T04 disponivel faria a fusao **descartar
        # silenciosamente** a quarta entrada -- o pior dos dois mundos: a tela
        # mostraria T04 e a decisao a ignoraria. Por isso a escolha segue a
        # disponibilidade real de T04, e nao uma constante.
        #
        # Que os dois deem praticamente o mesmo resultado (DeLong p = 0,182,
        # secao 4.12 de RESULTADOS.md) e um achado medido, nao uma licenca para
        # trocar um pelo outro sem criterio.
        try:
            if "T04" in self.techniques:
                fusion_path = WEIGHTS_DIR / "t05_fusion__quatro_fontes.pkl"
                if not fusion_path.exists():
                    fusion_path = WEIGHTS_DIR / "t05_fusion.pkl"
            else:
                fusion_path = WEIGHTS_DIR / "t05_fusion.pkl"

            if fusion_path.exists():
                self.techniques["T05"] = T05Fusion().load(fusion_path)
                self.fusion_model_name = fusion_path.name
            else:
                self.load_errors["T05"] = f"modelo ausente em {fusion_path}"
        except Exception as error:                      # noqa: BLE001
            self.load_errors["T05"] = str(error)

    # ------------------------------------------------------------------

    def analyze(
        self, image_path: Path, caminho_t02: Path | None = None
    ) -> dict[str, dict]:
        """Executa cada modulo isoladamente sobre uma unica imagem.

        ``image_path`` esta na condicao medida do Capitulo 4 -- 256x256 -- e e o
        que alimenta **todas** as fontes da fusao.

        ``caminho_t02``, quando presente, e a mesma imagem em resolucao nativa
        (com teto). A T02 e entao avaliada **duas vezes**: o escore nativo vai
        para a tela, porque e a condicao em que a tecnica foi verificada, e o de
        256 px segue alimentando a T05, porque foi nessa distribuicao que a
        fusao foi ajustada. Trocar a entrada da fusao aqui mediria o reajuste em
        vez da mudanca -- armadilha 6 de ESTADO_ATUAL.md.

        Omitir o argumento reproduz o comportamento anterior a 31/08/2026, que e
        o que ``scripts/medir_rnf01_com_t04.py`` mede.
        """
        outcomes: dict[str, dict] = {}
        source_scores: dict[str, np.ndarray] = {}

        for technique_id in ("T01", "T02", "T03", "T04"):
            technique = self.techniques.get(technique_id)
            if technique is None:
                outcomes[technique_id] = {
                    "status": "indisponivel",
                    "detail": self.load_errors.get(technique_id, "modulo nao carregado"),
                }
                continue
            try:
                started = time.perf_counter()
                nativo: float | None = None
                if technique_id == "T02" and caminho_t02 is not None:
                    # UMA chamada, dois escores. ``predict_proba`` ja opera
                    # em lote: escreve todos os caminhos num CSV e lanca UM
                    # subprocesso. Duas chamadas pagariam o arranque do
                    # interpretador e a carga do modelo duas vezes -- 51,16 s
                    # contra 21,49 s medidos em 31/08/2026, com os escores
                    # **identicos bit a bit** nas duas formas.
                    try:
                        par = technique.predict_proba([image_path, caminho_t02])
                        probability = float(par[0])     # 256 px -> fusao
                        nativo = float(par[1])          # nativa -> tela
                    except Exception as falha:          # noqa: BLE001
                        # O modo de falha esperado e falta de memoria na
                        # imagem grande. Juntar as duas passadas nao pode
                        # custar o escore de 256 px, que e o que a fusao
                        # precisa -- entao refaz so com ele (RN07).
                        probability = float(
                            technique.predict_proba([image_path])[0]
                        )
                        outcomes.setdefault(technique_id, {})["aviso"] = (
                            "resolucao nativa indisponivel "
                            f"({sanitize_for_table(falha, 90)}); "
                            "exibindo o escore de 256 px"
                        )
                else:
                    probability = float(technique.predict_proba([image_path])[0])
                source_scores[technique_id] = np.array([probability])
                aviso = outcomes.get(technique_id, {}).get("aviso")
                outcomes[technique_id] = {
                    "status": "ok",
                    "score": probability,
                    "elapsed": time.perf_counter() - started,
                }
                if aviso:
                    outcomes[technique_id]["aviso"] = aviso
                if nativo is not None:
                    # A fusao continua vendo os 256 px; a tela ve a nativa.
                    outcomes[technique_id]["score_fusao"] = probability
                    outcomes[technique_id]["score"] = nativo
                    outcomes[technique_id]["resolucao_nativa"] = True
            except (TechniqueError, Exception) as error:    # noqa: BLE001
                # RN07: registra a falha e prossegue com os demais modulos.
                outcomes[technique_id] = {"status": "erro", "detail": str(error)}

        fusion = self.techniques.get("T05")
        if fusion is None:
            outcomes["T05"] = {
                "status": "indisponivel",
                "detail": self.load_errors.get("T05", "modulo nao carregado"),
            }
        elif not source_scores:
            outcomes["T05"] = {
                "status": "indisponivel",
                "detail": "nenhuma tecnica-fonte produziu escore",
            }
        else:
            try:
                started = time.perf_counter()
                matrix = build_score_matrix(source_scores, 1)
                probability = float(fusion.predict_proba_scores(matrix)[0])
                outcomes["T05"] = {
                    "status": "ok",
                    "score": probability,
                    "elapsed": time.perf_counter() - started,
                    "n_sources": len(source_scores),
                }
            except Exception as error:                  # noqa: BLE001
                outcomes["T05"] = {"status": "erro", "detail": str(error)}

        return outcomes


# ---------------------------------------------------------------------------
# Validacao de entrada (RF06)
# ---------------------------------------------------------------------------


def validate_upload(path: str | None) -> tuple[bool, str]:
    """Valida formato e tamanho antes do envio ao pipeline.

    O formato e confirmado pelo conteudo do arquivo, e nao apenas pela
    extensao, de modo a rejeitar arquivos renomeados.
    """
    if not path:
        return False, "Nenhuma imagem foi enviada."

    file_path = Path(path)
    if not file_path.exists():
        return False, "Arquivo nao encontrado."

    size = file_path.stat().st_size
    if size > MAX_FILE_BYTES:
        return False, (
            f"Arquivo de {size / 1024 / 1024:.1f} MB excede o limite de 10 MB (RN02)."
        )

    if file_path.suffix.lower() not in ALLOWED_EXTENSIONS:
        return False, (
            f"Extensao '{file_path.suffix}' nao suportada. "
            f"Formatos aceitos: PNG, JPEG e WebP (RN01)."
        )

    try:
        with Image.open(file_path) as image:
            detected = (image.format or "").upper()
    except Exception:                                   # noqa: BLE001
        return False, "Arquivo corrompido ou nao reconhecido como imagem."

    if detected not in ALLOWED_FORMATS:
        return False, (
            f"O conteudo do arquivo e do tipo {detected}, nao suportado. "
            f"Formatos aceitos: PNG, JPEG e WebP (RN01)."
        )
    return True, "ok"


# ---------------------------------------------------------------------------
# Normalizacao do envio
# ---------------------------------------------------------------------------


@contextlib.contextmanager
def envio_normalizado(origem: Path) -> Iterator[Path]:
    """Poe o upload na mesma condicao em que as tecnicas foram medidas.

    Por que isto existe
    -------------------
    Todos os numeros do Capitulo 4 foram medidos sobre o corpus normalizado por
    ``scripts/normalize_corpus.py``: menor lado a 256 px com LANCZOS, recorte
    central, RGB. Ate 31/08/2026 a interface entregava o arquivo **cru** as
    cinco tecnicas, e um envio de 1024 px colocava T01, T03 e T04 fora da
    condicao em que foram treinadas e avaliadas.

    O efeito era visivel na tela. Com ``dalle3_00000`` (sintetica), o mesmo
    arquivo nas duas resolucoes:

        =======  ==============  ==================
        Tecnica     1024 px cru  256 px normaliz.
        =======  ==============  ==================
        T01         0,0% errado      99,5% correto
        T02       99,9% correto        0,0% errado
        =======  ==============  ==================

    T01 e T02 trocavam de lado exatamente. Nao e defeito de nenhuma das duas: a
    T01 foi treinada em 256 px e a T02 e espectral, medida em resolucao nativa
    por construcao. Normalizar acerta a T01 e piora a T02 na tela -- e a perda
    de 17,8 p.p. da secao 4.9, que e resultado medido, nao dano deste conserto.

    Efeito colateral util: uma imagem grande nunca mais chega crua aos modelos,
    o que elimina o caso de 214 s observado em 1024 px no processo frio.

    Custo
    -----
    Uma imagem ja conforme -- 256 x 256 em RGB, que e o caso do corpus inteiro
    -- e devolvida sem copia, para que ``scripts/medir_rnf01_com_t04.py`` siga
    medindo o mesmo caminho de antes. Sobra apenas a abertura do arquivo para
    conferir dimensoes e modo.

    RNF03: o arquivo normalizado vive num diretorio temporario removido na
    saida do contexto, mesmo em caso de excecao.
    """
    with Image.open(origem) as imagem:
        ja_conforme = imagem.size == (IMAGE_SIZE, IMAGE_SIZE) and imagem.mode == "RGB"
        # A conversao acontece dentro do ``with`` porque ``resize_and_center_crop``
        # le os pixels; o descritor precisa continuar aberto ate aqui.
        normalizada = None if ja_conforme else resize_and_center_crop(imagem, IMAGE_SIZE)
        if normalizada is not None:
            # ``crop`` e preguicoso no PIL. Forcar a leitura aqui garante que
            # nada dependa do descritor depois que o ``with`` o fechar.
            normalizada.load()

    if normalizada is None:
        # Fora do ``with``: o arquivo ja esta fechado quando as tecnicas o leem.
        # T02 abre a imagem num subprocesso e o Windows nao gosta de dois
        # descritores concorrentes sobre o mesmo arquivo.
        yield origem
        return

    with tempfile.TemporaryDirectory(prefix="tcc3_envio_") as pasta:
        # PNG: mesmo formato de saida da normalizacao do corpus, e sem
        # recompressao com perdas sobre o que sera analisado.
        destino = Path(pasta) / "envio_normalizado.png"
        normalizada.save(destino, format="PNG")
        normalizada.close()
        yield destino


@contextlib.contextmanager
def envio_para_t02(origem: Path) -> Iterator[Path | None]:
    """Entrega a T02 a maior resolucao que esta maquina aguenta.

    A T02 e literalmente *"Any-Resolution AI-Generated Image Detection by
    **Spectral** Learning"*: seu sinal e a distribuicao espectral, e reamostrar
    para 256x256 destroi a evidencia de alta frequencia que ela explora. A
    secao 4.9 mede isso -- media 17,8 p.p. abaixo do publicado --, e o controle
    do ``glide`` (nativo 256 px, intocado pela normalizacao) fecha a questao ao
    medir 0,9626 contra 0,902 publicado.

    Por que ha um teto
    ------------------
    Em 31/08/2026, a T02 sobre ``adobe_firefly_00002`` (2688x1536, 4,13 MPx)
    consumiu **5.931 MiB dos 6.144** da placa, com a interface parada. Com a
    interface no ar, a mesma chamada esgotou a memoria e **derrubou os dois
    servicos T04 do WSL2**. A RN02 aceita arquivos de 10 MB, que podem ser bem
    maiores que isso -- sem teto, um envio grande derruba a aplicacao.

    O teto vem da 4.9: nao ha dose-resposta, o efeito e de limiar -- reducao de
    2x fica dentro da tolerancia de 5 p.p., e so a partir de ~3,5x a degradacao
    satura. O padrao de 1.536 px no maior lado mantem a maioria dos envios
    dentro dessa faixa. **Nao e teto medido**; a medicao de ~200 imagens que o
    fixaria esta registrada como pergunta em aberto no ESTADO_ATUAL.md.

    Devolve ``None`` quando a politica esta desligada
    (``TCC3_T02_NATIVA=0``), caso em que a T02 volta a receber os 256 px como
    todas as demais.
    """
    if not TCC3_T02_NATIVA:
        yield None
        return

    with Image.open(origem) as imagem:
        largura, altura = imagem.size
        maior = max(largura, altura)
        if maior <= TETO_T02:
            # Ja cabe: entrega o proprio arquivo, sem reamostrar. Este e o caso
            # que preserva integralmente o sinal espectral da T02.
            reduzida = None
        else:
            escala = TETO_T02 / maior
            destino_tam = (max(1, round(largura * escala)), max(1, round(altura * escala)))
            reduzida = imagem.convert("RGB").resize(destino_tam, Image.LANCZOS)
            reduzida.load()

    if reduzida is None:
        yield origem
        return

    with tempfile.TemporaryDirectory(prefix="tcc3_t02_") as pasta:
        destino = Path(pasta) / "envio_t02.png"
        reduzida.save(destino, format="PNG")
        reduzida.close()
        yield destino


# ---------------------------------------------------------------------------
# Renderizacao
# ---------------------------------------------------------------------------


def sanitize_for_table(text: str, limit: int = 130) -> str:
    """Torna uma mensagem de erro segura para uma celula de tabela Markdown.

    Mensagens de erro trazem saida de subprocesso com quebras de linha e
    caminhos do Windows. Sem tratamento, cada quebra encerra a linha da tabela
    e o restante e renderizado como uma nova linha vazia, quebrando o layout.
    O caractere de barra vertical delimita colunas e tambem precisa ser
    neutralizado.
    """
    collapsed = " ".join(str(text).split())      # remove \n, \r e espacos repetidos
    collapsed = collapsed.replace("|", "\\|")
    if len(collapsed) > limit:
        collapsed = collapsed[: limit - 1].rstrip() + "…"
    return collapsed


def format_results(outcomes: dict[str, dict]) -> str:
    """Sumario textual consolidando nome e escore de cada tecnica (RF04).

    Nenhuma classificacao automatica ou valor agregado alem do escore de T05 --
    que e o proprio resultado da tecnica de fusao -- e apresentado (RN04).
    """
    lines = ["## Resultados por técnica", ""]
    lines.append("| Técnica | Método | Escore de síntese | Situação |")
    lines.append("|---|---|---|---|")

    for technique_id in DISPLAY_ORDER:            # RN05: ordem fixa
        outcome = outcomes.get(technique_id, {"status": "indisponivel", "detail": "-"})
        name = TECHNIQUE_NAMES.get(technique_id, technique_id)
        if outcome["status"] == "ok":
            situacao = f"concluída em {outcome['elapsed']:.2f} s"
            if outcome.get("resolucao_nativa"):
                # Deixa visivel que esta linha nao foi medida em 256 x 256.
                situacao += " · resolução nativa"
            if outcome.get("aviso"):
                situacao += f" · {sanitize_for_table(outcome['aviso'], 90)}"
            lines.append(
                f"| **{technique_id}** | {name} | **{outcome['score'] * 100:.1f}%** | "
                f"{situacao} |"
            )
        else:
            detail = sanitize_for_table(outcome.get("detail", ""))
            status = {"erro": "erro", "indisponivel": "indisponível"}.get(
                outcome["status"], outcome["status"]
            )
            lines.append(
                f"| **{technique_id}** | {name} | resultado não disponível | "
                f"{status}: {detail} |"
            )

    lines += [
        "",
        "> O escore indica a estimativa de síntese artificial produzida por cada "
        "técnica isoladamente. O sistema não emite classificação binária "
        "automática: a interpretação do conjunto de resultados cabe ao usuário "
        "(RN04).",
        "",
        f"> Antes da análise a imagem é normalizada para {IMAGE_SIZE} × {IMAGE_SIZE} "
        "— menor lado redimensionado com LANCZOS e recorte central —, que é a "
        "condição em que T01, T03 e T04 foram treinadas e avaliadas. O espectro "
        "exibido é o da imagem normalizada.",
    ]

    t02 = outcomes.get("T02", {})
    if t02.get("resolucao_nativa"):
        # Sem esta nota a tela mostraria dois números de T02 sem dizer que sao
        # dois, e o leitor tentaria fechar a conta da fusao com o errado.
        lines += [
            "",
            "> A **T02 é avaliada em resolução nativa**, e não em "
            f"{IMAGE_SIZE} × {IMAGE_SIZE}: seu sinal é espectral e o "
            "redimensionamento destrói a evidência de alta frequência que ela "
            "explora (seção 4.9). O escore exibido acima é o da resolução "
            "nativa. **A T05 continua sendo alimentada pelo escore de "
            f"{IMAGE_SIZE} × {IMAGE_SIZE}** "
            f"(**{t02.get('score_fusao', float('nan')) * 100:.1f}%**), que é a "
            "distribuição em que a fusão foi ajustada — por isso os dois "
            "números não fecham por soma direta.",
        ]
    return "\n".join(lines)


def build_interface(service: DetectionService | None = None) -> gr.Blocks:
    service = service or DetectionService()

    def run_analysis(image_path):
        valid, message = validate_upload(image_path)
        if not valid:
            return f"### Envio inválido\n\n{message}", None

        # A normalizacao envolve **as duas** saidas de proposito: os modulos e
        # o espectro exibido precisam ver a mesma imagem. O espectro sobre o
        # arquivo cru ainda espremia uma 1024x768 num quadrado de 256, o que
        # distorce a geometria justamente do que se pede ao usuario para ler.
        with envio_normalizado(Path(image_path)) as caminho, \
                envio_para_t02(Path(image_path)) as caminho_t02:
            # Envio ja conforme (256 x 256 em RGB): os dois gerenciadores
            # devolvem o mesmo arquivo, e a segunda passada da T02 seria
            # uma repeticao de ~40 s sem nenhuma informacao nova.
            if caminho_t02 == caminho:
                caminho_t02 = None
            outcomes = service.analyze(caminho, caminho_t02=caminho_t02)
            spectrum = magnitude_spectrum_array(caminho)          # RF05
        return format_results(outcomes), spectrum

    with gr.Blocks(title="Detecção de Imagens Geradas por IA") as demo:
        gr.Markdown(
            "# Detecção de Imagens Geradas por Inteligência Artificial\n\n"
            "Interface demonstrativa do TCC de Elian Ferreira (UNIVALI, 2026). "
            "Envie uma imagem em PNG, JPEG ou WebP com até 10 MB."
        )

        if service.load_errors:
            unavailable = ", ".join(sorted(service.load_errors))
            gr.Markdown(
                f"> **Módulos indisponíveis nesta sessão:** {unavailable}. "
                "As demais técnicas seguem operacionais (RNF05)."
            )

        with gr.Row():
            with gr.Column(scale=1):
                image_input = gr.Image(type="filepath", label="Imagem para análise", height=320)
                analyze_button = gr.Button("Analisar", variant="primary")
            with gr.Column(scale=1):
                spectrum_output = gr.Image(label="Espectro de magnitude", height=320)

        results_output = gr.Markdown()

        analyze_button.click(
            fn=run_analysis,
            inputs=image_input,
            outputs=[results_output, spectrum_output],
            show_progress="full",       # RNF06: indicacao visual de progresso
        )

    return demo


def main() -> None:
    # RNF03: nenhum arquivo intermediario e retido apos o encerramento; o
    # diretorio temporario da sessao e removido com o processo.
    with tempfile.TemporaryDirectory(prefix="tcc3_sessao_") as session_dir:
        import os

        os.environ["GRADIO_TEMP_DIR"] = session_dir
        build_interface().launch(show_api=False)


if __name__ == "__main__":
    main()
