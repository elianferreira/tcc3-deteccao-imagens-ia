"""T04 - Deteccao por geometria projetiva (Sarkar et al., 2024).

Replicacao do metodo publicado em https://github.com/hanlinm2/projective-geometry
(CVPR 2024), que detecta imagens sinteticas a partir de inconsistencias
geometricas, sem acesso direto aos pixels da imagem.

Sao extraidas tres representacoes geometricas independentes por meio de modelos
auxiliares oficiais, cada uma com seu proprio classificador:

===================  ==========================  ==================
Representacao        Classificador               Diretorio oficial
===================  ==========================  ==================
Campo de perspectiva ResNet-50                   perspective_fields/
Segmentos de reta    PointNet                    line_segment/
Objeto-sombra        ResNet-50                   object_shadow/
===================  ==========================  ==================

Como os classificadores operam apenas sobre caracteristicas geometricas
derivadas, e nao sobre os pixels, o metodo tende a ser mais robusto a variacoes
de baixo nivel especificas de cada gerador.

A tecnica e executada em modo de inferencia. Os tres escores sao agregados em
uma unica probabilidade, tal como no trabalho original; os escores individuais
permanecem acessiveis em ``last_component_scores_`` para a analise de erros
prevista na Etapa 5.

Disponibilidade parcial
-----------------------
Os classificadores oficiais nao recebem pixels: cada um consome uma
representacao geometrica ja extraida por um modelo auxiliar que depende de
detectron2, sem suporte em Windows. Por isso T04 constava como indisponivel.

Uma das tres representacoes deixou de estar bloqueada. O SSISv2 foi posto para
rodar no WSL2 (``documentacao/T04_AMBIENTE_WSL2.md``) e os mapas objeto-sombra do
corpus deste trabalho foram extraidos e classificados; os escores ficam em
``resultados/t04_escores_combined.csv``. Quando esse arquivo existe, T04 passa a
contribuir com **uma** das tres representacoes, e ``nanmean`` ignora as outras
duas -- o mesmo mecanismo previsto em RN07 para falha de modulo.

A consequencia deve ser dita onde o numero aparecer: o valor resultante nao e o
T04 do artigo, e sim seu componente objeto-sombra isolado. Reporta-se como
**T04 (objeto-sombra)**. Campos de perspectiva e segmentos de reta seguem sem
extrator; ver ``externo/CONTRATO.md``.
"""

from __future__ import annotations

import csv
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Sequence

import numpy as np

from ...configuracao import EXTERNAL, ExternalConfig
from ..base import BaseTechnique, TechniqueError

COMPONENTS = ("perspective_fields", "line_segment", "object_shadow")

# ---------------------------------------------------------------------------
# Servico residente no WSL2
# ---------------------------------------------------------------------------
# Ate 30/08/2026 T04 so pontuava imagens do corpus, por escores pre-extraidos.
# A interface, que recebe imagens arbitrarias, reportava indisponibilidade.
#
# O servico (``automacao/wsl/servico_t04.py``) mantem os tres extratores
# residentes no WSL2 e pontua uma imagem qualquer sob demanda. Sao **dois**
# processos porque a extracao em lote nao usou o mesmo dispositivo para as tres
# representacoes, e o PointNet dos autores exige CUDA_VISIBLE_DEVICES=""
# no processo inteiro; o cabecalho do servico detalha.
#
# Paridade medida contra os escores da dissertacao: campos e retas exatos ate
# 1e-16; objeto-sombra em 2,3e-04, residuo de cuDNN com lote 1 contra lote 128.
SERVICE_ENDPOINTS = {
    "http://127.0.0.1:8404": ("object_shadow",),
    "http://127.0.0.1:8405": ("perspective_fields", "line_segment"),
}
# Desligado por padrao: o servico e opcional e T04 volta a se declarar
# indisponivel sem ele, exatamente como antes. TCC3_T04_SERVICO=1 liga.
SERVICE_ENABLED = os.environ.get("TCC3_T04_SERVICO", "") not in ("", "0", "false")
SERVICE_TIMEOUT = int(os.environ.get("TCC3_T04_SERVICO_TIMEOUT", "120"))

# Escores de objeto-sombra ja extraidos, ativados **explicitamente** por
# TCC3_T04_ESCORES.
#
# Por que uma variavel de ambiente e nao a mera existencia do arquivo: os
# escores cobrem o corpus deste trabalho, e so ele. A interface recebe imagens
# arbitrarias do usuario, que jamais estarao nesse CSV -- deduzir
# disponibilidade da presenca do arquivo faria T04 se anunciar disponivel na
# tela e falhar em toda imagem enviada, o oposto do que RN07 exige. O modo
# pre-extraido pertence a campanha experimental, que o liga de proposito
# (ver automacao/pipeline_t04_fusao.py).
_ESCORES_ENV = os.environ.get("TCC3_T04_ESCORES")
DEFAULT_PRECOMPUTED = Path(_ESCORES_ENV) if _ESCORES_ENV else None


class T04ProjectiveGeometry(BaseTechnique):
    technique_id = "T04"
    trainable = False   # extratores e classificadores oficiais pre-treinados

    def __init__(
        self,
        device: str = "cuda",
        external: ExternalConfig = EXTERNAL,
        python_executable: str | None = None,
        components: Sequence[str] = COMPONENTS,
        aggregation: str = "mean",
        timeout: int = 3600,
        precomputed: Path | None = None,
        service_enabled: bool | None = None,
        service_timeout: int | None = None,
    ) -> None:
        super().__init__(device=device)
        self.external = external
        self.python_executable = python_executable or self._default_interpreter()
        self.components = tuple(components)
        self.aggregation = aggregation
        self.timeout = timeout
        self.precomputed = Path(precomputed) if precomputed else DEFAULT_PRECOMPUTED
        self.service_enabled = (SERVICE_ENABLED if service_enabled is None
                                else service_enabled)
        self.service_timeout = (SERVICE_TIMEOUT if service_timeout is None
                                else service_timeout)
        self.last_component_scores_: dict[str, np.ndarray] = {}
        self._precomputed_cache: dict[str, float] | None = None
        self._fitted = True

    # ------------------------------------------------------------------

    def _load_precomputed(self) -> dict[str, dict[str, float]]:
        """Escores ja calculados, por componente, indexados pelo nome base.

        Aceita dois formatos:

        * uma coluna por representacao (``object_shadow``,
          ``perspective_fields``, ``line_segment``), produzido por
          ``automacao/consolidar_escores_t04.py``;
        * o formato antigo, de coluna unica ``escore_t04``, que era so de
          objeto-sombra -- mantido para nao invalidar arquivos ja gerados.

        A chave e o nome sem extensao: as representacoes intermediarias sao
        gravadas com outras extensoes que a imagem de origem.

        Celula vazia permanece ausente, e nao vira zero. A distincao importa: o
        agregador usa ``nanmean``, de modo que uma representacao sem escore e
        ignorada em vez de puxar a media para baixo.
        """
        if self._precomputed_cache is None:
            cache: dict[str, dict[str, float]] = {}
            with self.precomputed.open(newline="", encoding="utf-8") as arquivo:
                leitor = csv.DictReader(arquivo)
                colunas = [c for c in (leitor.fieldnames or []) if c in COMPONENTS]
                antigo = not colunas and "escore_t04" in (leitor.fieldnames or [])

                for linha in leitor:
                    if antigo:
                        cache[linha["arquivo"]] = {"object_shadow": float(linha["escore_t04"])}
                        continue
                    valores = {}
                    for componente in colunas:
                        bruto = (linha.get(componente) or "").strip()
                        if bruto:
                            valores[componente] = float(bruto)
                    cache[linha["arquivo"]] = valores
            self._precomputed_cache = cache
        return self._precomputed_cache

    def _default_interpreter(self) -> str:
        repo = self.external.geometry_repo
        for candidate in (repo / ".venv" / "Scripts" / "python.exe", repo / ".venv" / "bin" / "python"):
            if candidate.exists():
                return str(candidate)
        return sys.executable

    # ------------------------------------------------------------------

    def is_available(self) -> tuple[bool, str]:
        # Caminho do servico residente, que tem precedencia: e o unico que
        # pontua uma imagem arbitraria, e por isso o unico que serve a
        # interface. Os escores pre-extraidos cobrem apenas o corpus.
        if self.service_enabled:
            vivos, ausentes = self._sondar_servico()
            if vivos and not ausentes:
                return True, (
                    "completa: as tres representacoes, pelo servico residente "
                    "no WSL2 (portas 8404 e 8405)"
                )
            if vivos:
                return True, (
                    f"parcial: {', '.join(sorted(vivos))} pelo servico residente; "
                    f"sem resposta em {', '.join(ausentes)}"
                )
            # Nenhum endpoint responde: cai para os caminhos abaixo, que e o
            # comportamento anterior ao servico existir.

        # Caminho dos escores pre-extraidos. Uma das tres representacoes --
        # objeto-sombra -- deixou de estar bloqueada: o SSISv2 roda no WSL2
        # (documentacao/T04_AMBIENTE_WSL2.md) e seus mapas ja foram extraidos para o
        # corpus deste trabalho. As outras duas seguem sem extrator, e a
        # agregacao por nanmean as ignora, conforme RN07.
        #
        # A tecnica passa a estar disponivel, mas o que ela mede nao e mais o
        # T04 do artigo: e um de seus tres componentes. Onde o numero for
        # reportado, deve aparecer como "T04 (objeto-sombra)".
        if self.precomputed is not None and self.precomputed.exists():
            return True, (
                "parcial: apenas a representacao objeto-sombra, a partir de "
                f"escores pre-extraidos em {self.precomputed.name}; campos de "
                "perspectiva e segmentos de reta seguem sem extrator"
            )

        repo = self.external.geometry_repo
        if not repo.exists():
            return False, (
                f"repositorio oficial ausente em {repo}; execute automacao/setup_external.py"
            )
        missing = [c for c in self.components if not (repo / c).exists()]
        if missing:
            return False, f"subdiretorios oficiais ausentes: {missing}"
        if not self.external.geometry_weights.exists():
            return False, (
                f"pesos dos classificadores ausentes em {self.external.geometry_weights}"
            )
        if not Path(self.python_executable).exists() and shutil.which(self.python_executable) is None:
            return False, f"interpretador nao encontrado: {self.python_executable}"

        # O bloqueio real de T04 nao e um arquivo faltando: os tres
        # classificadores oficiais nao recebem pixels. Cada um carrega uma
        # representacao geometrica ja extraida por um modelo auxiliar externo,
        # que o repositorio nao distribui. Verificado no codigo oficial:
        #
        #   perspective_fields/fields_dataset.py:31  torch.load(field_path)
        #       -> tensores .pt de PerspectiveFields (Jin et al.), via detectron2
        #   object_shadow/dataset.py:20-21           Image.open(shadow/object)
        #       -> mascaras de SSISv2, via detectron2/AdelaiDet
        #   line_segment/lines_dataset.py:22         image_path_to_lines[...]
        #       -> segmentos pre-extraidos por detector de retas
        #
        # O proprio artigo declara: "All three classifiers are denied access to
        # image pixels, and look only at derived geometric features."
        #
        # Reportar "script de inferencia ausente" seria enganoso: sugere que
        # escrever um adaptador resolveria, quando o que falta sao os
        # extratores. Ver externo/CONTRATO.md.
        return False, (
            "os classificadores oficiais nao operam sobre pixels: exigem "
            "representacoes geometricas pre-extraidas (campos de perspectiva, "
            "mascaras objeto-sombra, segmentos de reta) produzidas por modelos "
            "auxiliares dependentes de detectron2, sem suporte em Windows. "
            "Ver externo/CONTRATO.md"
        )

    def extract_features(self, paths: Sequence[Path]) -> np.ndarray:
        """Matriz (n, 3) com os escores das tres representacoes geometricas."""
        available, reason = self.is_available()
        if not available:
            raise TechniqueError(f"T04 indisponivel: {reason}")

        paths = [Path(p).resolve() for p in paths]

        # O servico pontua imagem arbitraria; os demais caminhos, nao. Quando
        # ele esta de pe, e ele que responde.
        if self.service_enabled:
            vivos, _ = self._sondar_servico()
            if vivos:
                return self._extrair_pelo_servico(paths)

        columns = []
        failures = []
        for component in self.components:
            try:
                columns.append(self._run_component(component, paths))
            except TechniqueError as error:
                # A falha de uma representacao nao invalida as demais; o metodo
                # segue com as disponiveis, em conformidade com RN07.
                failures.append(f"{component}: {error}")
                columns.append(np.full(len(paths), np.nan))

        matrix = np.stack(columns, axis=1)
        if np.isnan(matrix).all():
            raise TechniqueError("T04: todas as representacoes falharam\n" + "\n".join(failures))

        self.last_component_scores_ = {
            component: matrix[:, index] for index, component in enumerate(self.components)
        }
        return matrix

    # ------------------------------------------------------------------
    # Cliente do servico residente
    # ------------------------------------------------------------------

    @staticmethod
    def _para_wsl(caminho: Path) -> str:
        """``C:\\Users\\...`` -> ``/mnt/c/Users/...``.

        Mesma conversao que os extratores em lote aplicam. O servico vive no
        WSL2 e enxerga o disco do Windows por ``/mnt``.
        """
        texto = str(caminho).replace("\\", "/")
        if len(texto) > 1 and texto[1] == ":":
            return f"/mnt/{texto[0].lower()}{texto[2:]}"
        return texto

    def _sondar_servico(self) -> tuple[set[str], list[str]]:
        """(representacoes vivas, endpoints sem resposta).

        Barato de proposito: ``is_available`` e chamada a cada carga da
        interface, e um endpoint fora do ar nao pode custar segundos.
        """
        import urllib.error
        import urllib.request

        vivas: set[str] = set()
        ausentes: list[str] = []
        for base, esperadas in SERVICE_ENDPOINTS.items():
            try:
                with urllib.request.urlopen(f"{base}/saude", timeout=2) as resposta:
                    saude = json.loads(resposta.read())
                vivas.update(saude.get("representacoes", esperadas))
            except (urllib.error.URLError, TimeoutError, OSError, ValueError):
                ausentes.append(base)
        return vivas, ausentes

    def _pontuar_pelo_servico(self, caminho: Path) -> dict:
        """Consulta os dois endpoints e funde as respostas."""
        import urllib.request

        fundido: dict = {}
        alvo = self._para_wsl(caminho)
        for base in SERVICE_ENDPOINTS:
            pedido = urllib.request.Request(
                f"{base}/escore",
                data=json.dumps({"caminho": alvo}).encode("utf-8"),
                headers={"Content-Type": "application/json"},
                method="POST")
            try:
                with urllib.request.urlopen(pedido, timeout=self.service_timeout) as r:
                    fundido.update(json.loads(r.read()))
            except Exception as erro:                       # noqa: BLE001
                # Um endpoint fora nao invalida o outro: as representacoes que
                # ele serviria ficam NaN e o nanmean as ignora (RN07).
                fundido.setdefault("falhas", {})[base] = f"{type(erro).__name__}: {erro}"
        return fundido

    def _extrair_pelo_servico(self, paths: Sequence[Path]) -> np.ndarray:
        """Matriz (n, 3) pelo servico residente, na ordem de ``self.components``."""
        matrix = np.full((len(paths), len(self.components)), np.nan, dtype=np.float64)
        falhas: list[str] = []

        for linha, caminho in enumerate(paths):
            resposta = self._pontuar_pelo_servico(caminho)
            for coluna, componente in enumerate(self.components):
                valor = resposta.get(componente)
                if valor is not None:
                    matrix[linha, coluna] = float(valor)
            if resposta.get("falhas"):
                falhas.append(f"{caminho.name}: {resposta['falhas']}")

        if np.isnan(matrix).all():
            raise TechniqueError(
                "T04: o servico nao produziu nenhum escore\n" + "\n".join(falhas[:5]))

        self.last_component_scores_ = {
            componente: matrix[:, indice]
            for indice, componente in enumerate(self.components)
        }
        return matrix

    def _run_component(self, component: str, paths: Sequence[Path]) -> np.ndarray:
        """Escores de uma representacao geometrica.

        Objeto-sombra vem dos escores pre-extraidos quando disponiveis; as
        demais seguem pelo contrato de ``infer.py`` descrito abaixo, que
        continua sem extrator.
        """
        if self.precomputed is not None and self.precomputed.exists():
            cache = self._load_precomputed()

            # Duas ausencias distintas, com tratamentos distintos.
            #
            # Imagem fora do arquivo: nunca foi extraida, e pedir escore para
            # ela e erro do chamador -- falha alto, com instrucao de como
            # corrigir.
            desconhecidas = [p.stem for p in paths if p.stem not in cache]
            if desconhecidas:
                raise TechniqueError(
                    f"{component}: {len(desconhecidas)} imagens fora dos escores "
                    f"pre-extraidos (ex.: {desconhecidas[:3]}). Rode a extracao "
                    "para este subconjunto e consolide com "
                    "automacao/consolidar_escores_t04.py"
                )

            # Imagem presente, mas sem esta representacao: ausencia legitima --
            # um extrator pode ter rodado sobre um subconjunto menor que outro.
            # Vira NaN, que ``nanmean`` ignora, em vez de derrubar a coluna
            # inteira por causa de uma imagem.
            return np.asarray([cache[p.stem].get(component, np.nan) for p in paths],
                              dtype=np.float64)

        return self._run_component_oficial(component, paths)

    def _run_component_oficial(self, component: str, paths: Sequence[Path]) -> np.ndarray:
        """Executa o classificador oficial de uma representacao geometrica.

        A invocacao segue o contrato documentado em ``externo/CONTRATO.md``:
        um script ``infer.py`` no diretorio da representacao que recebe um CSV
        de caminhos e grava um JSON ``{caminho: escore}``.
        """
        script = self.external.geometry_repo / component / "infer.py"
        if not script.exists():
            raise TechniqueError(
                f"script de inferencia ausente: {script}. Consulte externo/CONTRATO.md "
                "para adaptar o script oficial ao contrato esperado."
            )

        with tempfile.TemporaryDirectory(prefix=f"geo_{component}_") as workdir:
            work = Path(workdir)
            input_csv = work / "input.csv"
            output_json = work / "scores.json"

            with input_csv.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.writer(handle)
                writer.writerow(["image"])
                for path in paths:
                    writer.writerow([str(path)])

            command = [
                self.python_executable, str(script),
                "--input", str(input_csv),
                "--output", str(output_json),
                "--weights", str(self.external.geometry_weights / component),
                "--device", self.device,
            ]
            completed = subprocess.run(
                command,
                cwd=str(self.external.geometry_repo),
                capture_output=True,
                text=True,
                timeout=self.timeout,
                check=False,
            )
            if completed.returncode != 0:
                tail = (completed.stderr or completed.stdout or "").strip()[-800:]
                raise TechniqueError(f"falha na execucao\n{tail}")
            if not output_json.exists():
                raise TechniqueError(f"saida nao gerada em {output_json}")

            scores = json.loads(output_json.read_text(encoding="utf-8"))

        by_name = {Path(key).name: float(value) for key, value in scores.items()}
        missing = [p.name for p in paths if p.name not in by_name]
        if missing:
            raise TechniqueError(f"{len(missing)} imagens sem escore (ex.: {missing[:3]})")
        return np.asarray([by_name[p.name] for p in paths], dtype=np.float64)

    def fit(self, paths: Sequence[Path], y: np.ndarray) -> "T04ProjectiveGeometry":
        """No-op: T04 utiliza os extratores e classificadores oficiais."""
        self._fitted = True
        return self

    # ------------------------------------------------------------------

    def predict_proba(self, paths: Sequence[Path]) -> np.ndarray:
        matrix = self.extract_features(paths)
        matrix = self._to_probabilities(matrix)

        if self.aggregation == "mean":
            # nanmean ignora representacoes indisponiveis para cada amostra.
            aggregated = np.nanmean(matrix, axis=1)
        elif self.aggregation == "max":
            aggregated = np.nanmax(matrix, axis=1)
        else:
            raise ValueError(f"agregacao desconhecida: {self.aggregation}")

        # Amostras sem nenhuma representacao valida recebem 0,5, valor neutro
        # que nao favorece nenhuma das classes.
        return np.where(np.isnan(aggregated), 0.5, aggregated)

    @staticmethod
    def _to_probabilities(matrix: np.ndarray) -> np.ndarray:
        """Normaliza cada coluna para [0, 1], preservando a ordenacao."""
        output = matrix.copy()
        for index in range(output.shape[1]):
            column = output[:, index]
            finite = column[np.isfinite(column)]
            if finite.size and (finite.min() < 0.0 or finite.max() > 1.0):
                output[:, index] = 1.0 / (1.0 + np.exp(-column))
        return output
