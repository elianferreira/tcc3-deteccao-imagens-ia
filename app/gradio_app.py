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

import sys
import tempfile
import time
from pathlib import Path

import gradio as gr
import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import EXTERNAL, TECHNIQUE_NAMES, WEIGHTS_DIR   # noqa: E402
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


class DetectionService:
    """Carrega os modelos uma unica vez e executa os modulos isoladamente.

    Os modelos sao carregados em memoria na inicializacao da interface; erros
    de inferencia em um modulo especifico sao capturados e reportados ao
    usuario sem interromper a execucao dos demais modulos.
    """

    def __init__(self) -> None:
        self.techniques: dict[str, object] = {}
        self.load_errors: dict[str, str] = {}
        self._load_all()

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
        try:
            fusion_path = WEIGHTS_DIR / "t05_fusion.pkl"
            if fusion_path.exists():
                self.techniques["T05"] = T05Fusion().load(fusion_path)
            else:
                self.load_errors["T05"] = f"modelo ausente em {fusion_path}"
        except Exception as error:                      # noqa: BLE001
            self.load_errors["T05"] = str(error)

    # ------------------------------------------------------------------

    def analyze(self, image_path: Path) -> dict[str, dict]:
        """Executa cada modulo isoladamente sobre uma unica imagem."""
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
                probability = float(technique.predict_proba([image_path])[0])
                source_scores[technique_id] = np.array([probability])
                outcomes[technique_id] = {
                    "status": "ok",
                    "score": probability,
                    "elapsed": time.perf_counter() - started,
                }
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
            lines.append(
                f"| **{technique_id}** | {name} | **{outcome['score'] * 100:.1f}%** | "
                f"concluída em {outcome['elapsed']:.2f} s |"
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
    ]
    return "\n".join(lines)


def build_interface(service: DetectionService | None = None) -> gr.Blocks:
    service = service or DetectionService()

    def run_analysis(image_path):
        valid, message = validate_upload(image_path)
        if not valid:
            return f"### Envio inválido\n\n{message}", None

        outcomes = service.analyze(Path(image_path))
        spectrum = magnitude_spectrum_array(Path(image_path))     # RF05
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
