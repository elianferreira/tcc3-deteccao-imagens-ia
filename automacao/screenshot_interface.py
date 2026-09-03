"""Captura a interface em execucao, para documentacao da monografia.

Sobe um navegador headless contra a interface Gradio ja em execucao, submete
uma imagem real do corpus, aguarda a analise concluir e grava duas capturas:
a tela inicial e a tela com os resultados.

As figuras resultantes documentam a Etapa 6 no Capitulo 4.

Uso::

    python interface/gradio_app.py                     # em um terminal
    python automacao/screenshot_interface.py --image data/corvi2024/fake/glide/glide_00000.png
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from codigo.configuracao import RESULTS_DIR      # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Captura a interface Gradio em execucao")
    parser.add_argument("--url", default="http://127.0.0.1:7860")
    parser.add_argument("--image", type=Path, required=True, help="imagem a submeter")
    parser.add_argument("--output", type=Path, default=RESULTS_DIR / "figuras" / "interface")
    parser.add_argument("--width", type=int, default=1440)
    parser.add_argument("--height", type=int, default=1000)
    parser.add_argument("--timeout", type=int, default=180_000,
                        help="tempo maximo de espera pela analise, em ms")
    args = parser.parse_args()

    if not args.image.exists():
        print(f"ERRO: imagem nao encontrada: {args.image}")
        return 1

    from playwright.sync_api import sync_playwright

    output_dir = args.output.parent
    output_dir.mkdir(parents=True, exist_ok=True)
    initial = args.output.with_name(args.output.name + "_inicial.png")
    result = args.output.with_name(args.output.name + "_resultado.png")

    with sync_playwright() as playwright:
        browser = playwright.chromium.launch()
        page = browser.new_page(viewport={"width": args.width, "height": args.height})

        print(f"abrindo {args.url} ...")
        page.goto(args.url, wait_until="networkidle", timeout=60_000)
        # O Gradio monta os componentes apos o carregamento inicial do bundle.
        page.wait_for_selector("text=Analisar", timeout=60_000)
        page.screenshot(path=str(initial), full_page=True)
        print(f"tela inicial gravada em {initial}")

        print(f"submetendo {args.image.name} ...")
        page.set_input_files("input[type=file]", str(args.image.resolve()))
        # Aguarda a miniatura aparecer antes de disparar a analise.
        page.wait_for_timeout(2500)
        page.click("text=Analisar")

        # A analise termina quando a tabela de resultados e renderizada. O
        # seletor casa apenas o prefixo sem acento para nao depender da grafia
        # exata do titulo.
        page.wait_for_selector("h2:has-text('Resultados por')", timeout=args.timeout)
        page.wait_for_timeout(1500)      # deixa o espectro terminar de pintar
        page.screenshot(path=str(result), full_page=True)
        print(f"tela de resultado gravada em {result}")

        browser.close()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
