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

import os
import sys
import tempfile
import time
from pathlib import Path

import gradio as gr
import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from codigo.configuracao import (                                       # noqa: E402
    EXTERNAL, IMAGE_SIZE, TECHNIQUE_NAMES, WEIGHTS_DIR,
)
from codigo.graficos import magnitude_spectrum_array                    # noqa: E402
# O pre-processamento vive em codigo/preprocessamento/. Reexportado aqui
# porque a interface e quem o aciona, e os testes de regressao do item 1
# entram por este modulo.
from codigo.preprocessamento.envio_interface import (                   # noqa: E402
    envio_normalizado, envio_para_t02,
)
from codigo.preprocessamento.validacao_envio import (                   # noqa: E402
    ALLOWED_EXTENSIONS, ALLOWED_FORMATS, MAX_FILE_BYTES, validate_upload,
)
from codigo.tecnicas.base import TechniqueError                         # noqa: E402
from codigo.tecnicas.t02_spai import T02SPAI                            # noqa: E402
from codigo.tecnicas.t03_benford import T03Benford                      # noqa: E402
from codigo.tecnicas.t04_geometria import T04ProjectiveGeometry         # noqa: E402
from codigo.tecnicas.t05_fusao import T05Fusion, build_score_matrix     # noqa: E402

DISPLAY_ORDER = ("T01", "T02", "T03", "T04", "T05")   # RN05


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
            from codigo.tecnicas.t01_coocorrencia import T01Cooccurrence

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
        o que ``automacao/medir_rnf01_com_t04.py`` mede.
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
