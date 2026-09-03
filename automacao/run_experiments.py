"""Etapas 4 e 5 - Campanha experimental e analise comparativa.

Executa os protocolos definidos na Secao 3.6.1 e grava metricas, figuras e
probabilidades brutas em ``resultados/``.

Uso::

    # Protocolo padrao (in-distribution), treinando T01, T03 e T05
    python automacao/run_experiments.py --protocol standard --fit

    # Generalizacao para geradores nao vistos
    python automacao/run_experiments.py --protocol ood --manifest data/manifesto_ood.csv --fit

    # Robustez, sobre subamostra do conjunto de teste
    python automacao/run_experiments.py --protocol robustness --limit 2000

    # Tres inicializacoes de T01 (sementes 42, 123 e 456)
    python automacao/run_experiments.py --protocol standard --multi-seed
"""

from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np                                          # noqa: E402

from codigo.configuracao import DATA_DIR, RESULTS_DIR, WEIGHTS_DIR, ensure_dirs   # noqa: E402
from codigo.experimentos.runner import ExperimentRunner         # noqa: E402
from codigo.graficos import (                                     # noqa: E402
    plot_auc_per_generator, plot_robustness_heatmap, plot_roc_curves,
    plot_training_history,
)
from codigo.sementes import set_deterministic                     # noqa: E402
from codigo.tecnicas.t02_spai import T02SPAI                 # noqa: E402
from codigo.tecnicas.t03_benford import T03Benford           # noqa: E402
from codigo.tecnicas.t04_geometria import T04ProjectiveGeometry   # noqa: E402
from codigo.tecnicas.t05_fusao import T05Fusion             # noqa: E402


def build_techniques(selection: list[str], device: str) -> dict:
    """Instancia apenas as tecnicas solicitadas.

    A instanciacao e tolerante a falhas: uma tecnica que nao possa ser
    construida -- por dependencia ausente, por exemplo -- e omitida, e as
    demais seguem disponiveis (RN07).
    """
    techniques: dict = {}

    if "T01" in selection:
        try:
            from codigo.tecnicas.t01_coocorrencia import T01Cooccurrence
            techniques["T01"] = T01Cooccurrence(device=device)
        except Exception as error:                          # noqa: BLE001
            print(f"[aviso] T01 indisponivel: {error}")

    if "T02" in selection:
        techniques["T02"] = T02SPAI(device=device)
    if "T03" in selection:
        techniques["T03"] = T03Benford()
    if "T04" in selection:
        techniques["T04"] = T04ProjectiveGeometry(device=device)
    if "T05" in selection:
        techniques["T05"] = T05Fusion()

    return techniques


MODEL_FILENAMES = {
    "T01": "t01_cooccurrence.pt",
    "T03": "t03_benford.pkl",
    "T05": "t05_fusion.pkl",
}


def save_models(techniques: dict) -> None:
    WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
    for technique_id, filename in MODEL_FILENAMES.items():
        technique = techniques.get(technique_id)
        if technique is None or not getattr(technique, "_fitted", False):
            continue
        try:
            technique.save(WEIGHTS_DIR / filename)
            print(f"  modelo de {technique_id} gravado em {WEIGHTS_DIR / filename}")
        except Exception as error:                          # noqa: BLE001
            print(f"  [aviso] falha ao gravar {technique_id}: {error}")


def load_models(techniques: dict) -> bool:
    """Carrega os modelos treinados em execucao anterior.

    Necessario nos protocolos executados sem ``--fit`` -- tipicamente OOD e
    robustez, que reaproveitam os modelos ajustados no protocolo padrao. T02 e
    T04 usam modelos oficiais e nao aparecem aqui.

    Retorna True se ao menos um modelo foi carregado.
    """
    loaded = False
    for technique_id, filename in MODEL_FILENAMES.items():
        technique = techniques.get(technique_id)
        if technique is None:
            continue
        path = WEIGHTS_DIR / filename
        if not path.exists():
            print(f"  [aviso] {technique_id}: modelo ausente em {path}")
            continue
        try:
            technique.load(path)
            print(f"  {technique_id}: carregado de {path}")
            loaded = True
        except Exception as error:                          # noqa: BLE001
            print(f"  [aviso] falha ao carregar {technique_id}: {error}")
    return loaded


def generate_figures(runner: ExperimentRunner, protocol: str, figures_dir: Path) -> None:
    """Gera as figuras da Etapa 5.

    A falha na geracao de uma figura nao invalida a campanha experimental, cujas
    metricas ja foram gravadas: o erro e reportado e as demais figuras seguem.
    """
    figures_dir.mkdir(parents=True, exist_ok=True)

    successful = [r for r in runner.results if not r.failed and r.probabilities is not None]
    if not successful:
        print("[aviso] nenhum resultado bem-sucedido; figuras nao geradas")
        return

    if protocol in ("standard", "ood"):
        clean = [r for r in successful if r.condition == "clean" and r.protocol == protocol]
        if clean:
            _, y_true, _ = runner._load_split("test")
            probabilities = {r.technique_id: r.probabilities for r in clean}
            try:
                plot_roc_curves(
                    y_true, probabilities, figures_dir / f"roc_{protocol}",
                    title=f"Curvas ROC - protocolo {protocol}",
                )
            except ValueError as error:
                print(f"  [aviso] curvas ROC nao geradas: {error}")

            auc_by_technique = {
                r.technique_id: {g: m["auc"] for g, m in r.per_generator.items()}
                for r in clean if r.per_generator
            }
            if auc_by_technique:
                try:
                    plot_auc_per_generator(
                        auc_by_technique, figures_dir / f"auc_por_gerador_{protocol}",
                        title=f"AUC por gerador - protocolo {protocol}",
                    )
                except ValueError as error:
                    print(f"  [aviso] AUC por gerador nao gerada: {error}")
            print(f"  figuras gravadas em {figures_dir}")

    if protocol == "robustness":
        auc_by_condition: dict[str, dict[str, float]] = {}
        for result in successful:
            auc_by_condition.setdefault(result.technique_id, {})[result.condition] = \
                result.metrics.get("auc", float("nan"))
        if auc_by_condition:
            plot_robustness_heatmap(auc_by_condition, figures_dir / "robustez_heatmap")
            print(f"  heatmap de robustez gravado em {figures_dir}")

    technique = runner.techniques.get("T01")
    if technique is not None and getattr(technique, "history", None):
        plot_training_history(technique.history, figures_dir / "treinamento_t01")


def report_multiseed(results: list) -> None:
    """Media e desvio padrao entre as inicializacoes aleatorias."""
    aucs = [r.metrics["auc"] for r in results if not r.failed]
    accuracies = [r.metrics["accuracy"] for r in results if not r.failed]
    if not aucs:
        return
    print("\n" + "=" * 60)
    print(f"T01 - {len(aucs)} inicializacoes aleatorias")
    print(f"  AUC ......... {np.mean(aucs):.4f} +/- {np.std(aucs, ddof=1 if len(aucs) > 1 else 0):.4f}")
    print(f"  Acuracia .... {np.mean(accuracies):.4f} +/- "
          f"{np.std(accuracies, ddof=1 if len(accuracies) > 1 else 0):.4f}")
    print("=" * 60)


def main() -> int:
    parser = argparse.ArgumentParser(description="Executa a campanha experimental")
    parser.add_argument("--protocol", choices=["standard", "ood", "robustness"], default="standard")
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--techniques", nargs="+", default=["T01", "T02", "T03", "T04", "T05"])
    parser.add_argument("--device", default=None, help="cuda ou cpu; padrao: detecta automaticamente")
    parser.add_argument("--fit", action="store_true", help="treina as tecnicas treinaveis antes de avaliar")
    parser.add_argument("--refit-fusion", action="store_true",
                        help="carrega T01/T03 do disco e reajusta apenas T05 sobre os "
                             "escores de validacao de todas as tecnicas disponiveis")
    parser.add_argument("--fusion-split", default="val", choices=["val", "fusion"],
                        help="particao usada para calibrar T05. 'val' reproduz a "
                             "calibracao in-distribution original; 'fusion' usa o "
                             "conjunto reservado com gerador nao visto "
                             "(ver documentacao/DECISOES_METODOLOGICAS.md)")
    parser.add_argument("--multi-seed", action="store_true",
                        help="repete o treinamento de T01 com as sementes 42, 123 e 456")
    parser.add_argument("--limit", type=int, default=None,
                        help="subamostra o conjunto de teste no protocolo de robustez")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-cache", action="store_true", help="desativa o cache de escores")
    args = parser.parse_args()

    ensure_dirs()
    set_deterministic(args.seed)

    if args.device is None:
        try:
            import torch
            args.device = "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            args.device = "cpu"
    print(f"Dispositivo: {args.device}")

    manifest = args.manifest or (
        DATA_DIR / ("manifesto_ood.csv" if args.protocol == "ood" else "manifesto_standard.csv")
    )
    if not manifest.exists():
        print(f"ERRO: manifesto nao encontrado em {manifest}")
        print("Execute antes: python automacao/prepare_dataset.py --corpus <caminho>")
        return 1

    techniques = build_techniques(args.techniques, args.device)
    print(f"Tecnicas registradas: {sorted(techniques)}\n")

    runner = ExperimentRunner(manifest, techniques, cache_scores=not args.no_cache)

    if args.fit:
        print("--- Ajuste das tecnicas ---")
        status = runner.fit_techniques(seed=args.seed)
        for technique_id, error in status.items():
            print(f"  {technique_id}: {'ok' if error is None else f'FALHOU ({error})'}")

        print(f"\n--- Ajuste de T05 (particao '{args.fusion_split}') ---")
        fusion_error = runner.fit_fusion(protocol=args.protocol, split=args.fusion_split)
        print(f"  T05: {'ok' if fusion_error is None else f'FALHOU ({fusion_error})'}")

        print("\n--- Persistencia dos modelos ---")
        save_models(techniques)
    else:
        # Sem --fit, reaproveita os modelos ajustados em execucao anterior.
        print("--- Carregamento dos modelos treinados ---")
        trainable = [t for t in args.techniques if t in MODEL_FILENAMES]
        # T05 e reajustado adiante quando --refit-fusion for usado; sua ausencia
        # em disco nao deve impedir o carregamento das demais.
        required = [t for t in trainable if not (args.refit_fusion and t == "T05")]
        if required and not load_models(techniques):
            print(
                "\nERRO: nenhum modelo treinado encontrado em "
                f"{WEIGHTS_DIR}.\nExecute antes o protocolo padrao com --fit:\n"
                "  python automacao/run_experiments.py --protocol standard --fit"
            )
            return 1

        if args.refit_fusion:
            # Reajusta a fusao sobre os escores de validacao produzidos por
            # TODAS as tecnicas agora disponiveis. Necessario sempre que o
            # conjunto de fontes muda: um T05 ajustado quando so uma tecnica
            # existia aprende a decidir por ela e ignora as demais mesmo depois
            # que passam a responder.
            print(f"\n--- Reajuste de T05 sobre a particao '{args.fusion_split}' ---")
            fusion_error = runner.fit_fusion(protocol=args.protocol, split=args.fusion_split)
            print(f"  T05: {'ok' if fusion_error is None else f'FALHOU ({fusion_error})'}")
            if fusion_error is None:
                fusion = techniques.get("T05")
                if fusion is not None:
                    fusion.save(WEIGHTS_DIR / MODEL_FILENAMES["T05"])
                    print(f"  pesos: {WEIGHTS_DIR / MODEL_FILENAMES['T05']}")
                    weights = fusion.contribution_weights()
                    print("  peso por dominio de evidencia:")
                    for name, value in weights.items():
                        print(f"    {name}: {value:+.4f}")

    print(f"\n--- Protocolo: {args.protocol} ---")
    if args.protocol == "robustness":
        with tempfile.TemporaryDirectory(prefix="robustez_") as workdir:
            runner.run_robustness(Path(workdir), limit=args.limit)
    elif args.multi_seed:
        results = runner.run_multi_seed("T01")
        report_multiseed(results)
        runner.run_protocol(args.protocol)
    else:
        runner.run_protocol(args.protocol)

    print("\n--- Resultados ---")
    for result in runner.results:
        if result.failed:
            print(f"  {result.technique_id} [{result.condition}]: ERRO - {result.error[:70]}")
        else:
            print(f"  {result.technique_id} [{result.condition}]: "
                  f"AUC={result.metrics['auc']:.4f}  "
                  f"Acc={result.metrics['accuracy']:.4f}  "
                  f"F1={result.metrics['f1']:.4f}  "
                  f"{result.metrics.get('ms_per_image', float('nan')):.1f} ms/img")

    paths = runner.save(prefix=f"resultados_{args.protocol}")
    print(f"\nMetricas gravadas em {paths['summary']}")

    generate_figures(runner, args.protocol, RESULTS_DIR / "figuras")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
