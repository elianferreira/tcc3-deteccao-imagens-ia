"""Extrai as tres representacoes da T04 sobre as 999 imagens do RAISE-1k.

Por que esta extracao existe
----------------------------
A medicao de 07/09/2026 que sustenta a A3 -- "a T05 vai de 4,3% para 47,4% de
falso positivo sob troca de fonte real" -- foi feita com a **T04 como NaN**, ou
seja, sobre a fusao de **tres** fontes, nao a de quatro que o sistema entrega.
O motivo registrado em `fpr_raise1k_todas.py:28` foi "bloqueio externo dos
extratores".

⛔ **Esse motivo ja estava vencido quando foi dado.** A sessao de 07/09 registra,
no mesmo dia, que o bloqueio tinha sido resolvido em 30/08: os servicos
residentes do WSL2 pontuam imagem arbitraria sob demanda. O que faltava era
ambiente de extracao, e ele existe desde entao.

E a consequencia nao e so aritmetica. A contribuicao que sobrou para o trabalho
e *"so a tecnica de pesos congelados atravessa a troca de fonte real"*, e hoje
ela se apoia em **uma unica instancia**, a T02. Mas a **T04 tambem e de pesos
oficiais dos autores** -- nao e treinada aqui. Ela e a segunda instancia da
tese, e ficou de fora justamente da medicao que criou a tese.

O que falta em disco, e o que nao falta
----------------------------------------
`resultados/t04_escores_componentes_ood_completo.csv` cobre 72.637 imagens do
corpus, incluindo 29.999 do COCO — e **nao serve para esta medicao**.

⛔ **Conferido em 23/09, e o cache nao cobre o controle.** As 999 reais do
controle sao sorteadas de `manifesto_ood13.csv`, que aponta para
`data/corvi2024_escala`, e os nomes de la passam de `coco_058406` — acima dos
`coco_029999` do corpus em cache. Parear por nome teria casado imagens
diferentes ou falhado em silencio (armadilha nº 5).

Por isso a rodada extrai **os dois conjuntos**, 999 cada:

    raise1k   data/raise1k_norm/real/raise1k        a fonte real nova
    coco      as mesmas 999 de `imagens_coco()`     o controle, semente 42

Usar o mesmo controle da rodada de 07/09 e o que mantem as duas comparaveis.
~5 s por imagem com os modelos quentes (8,8 s a frio), ~2,8 h no total.

Formato de saida
----------------
Igual ao do cache, para o consumidor nao precisar saber de onde veio:

    arquivo,split,label,object_shadow,perspective_fields,line_segment,n_retas

`n_retas` e extra e existe por uma razao: acima de **250 retas** o amostrador do
PointNet entra no ramo em que o escore depende da amostragem e deixa de ser
reproduzivel. No corpus a mediana e 49-81; a primeira imagem do RAISE deu 94.
A coluna permite conferir em vez de supor.

Retomada
--------
⚠️ Processos longos deste projeto ja morreram cinco vezes sem traceback. O CSV e
aberto em `append` e descarregado a cada linha; a proxima execucao pula o que ja
esta gravado. Uma falha por imagem **nao** derruba a rodada: a coluna vai vazia,
que o `nanmean` do agregador ja trata como ausencia.

Uso::

    python automacao/extrair_t04_raise1k.py --conjunto raise1k
    python automacao/extrair_t04_raise1k.py --conjunto coco
    python automacao/extrair_t04_raise1k.py --conjunto raise1k --limite 5
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ))
RAISE_DIR = RAIZ / "data" / "raise1k_norm" / "real" / "raise1k"
SAIDAS = {
    "raise1k": RAIZ / "resultados" / "t04_escores_raise1k.csv",
    "coco": RAIZ / "resultados" / "t04_escores_coco_controle.csv",
}

# Os dois servicos, e o que cada um responde. A divisao nao e arbitraria: o
# PointNet das retas exige CUDA_VISIBLE_DEVICES="" no processo inteiro, o que
# conflita com o objeto-sombra em GPU (armadilha nº 11).
SERVICOS = (
    ("http://127.0.0.1:8404/escore", ("object_shadow",)),
    ("http://127.0.0.1:8405/escore", ("perspective_fields", "line_segment")),
)
LIMITE_RETAS = 250
CAMPOS = ("arquivo", "split", "label", "object_shadow",
          "perspective_fields", "line_segment", "n_retas")


def caminho_wsl(caminho: Path) -> str:
    """C:\\Users\\... -> /mnt/c/Users/... — os servicos vivem do lado do WSL2."""
    texto = str(caminho.resolve()).replace("\\", "/")
    if len(texto) > 1 and texto[1] == ":":
        return f"/mnt/{texto[0].lower()}{texto[2:]}"
    return texto


def consultar(url: str, caminho: Path, tempo_limite: int) -> dict:
    corpo = json.dumps({"caminho": caminho_wsl(caminho)}).encode("utf-8")
    pedido = urllib.request.Request(
        url, data=corpo, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(pedido, timeout=tempo_limite) as resposta:
        return json.loads(resposta.read().decode("utf-8"))


def escorear(caminho: Path, tempo_limite: int) -> tuple[dict, list[str]]:
    """Consulta os dois servicos. Falha de um nao impede o outro."""
    valores: dict[str, object] = {}
    avisos: list[str] = []
    for url, chaves in SERVICOS:
        try:
            resposta = consultar(url, caminho, tempo_limite)
        except (urllib.error.URLError, TimeoutError, OSError) as erro:
            avisos.append(f"{url.rsplit('/', 2)[-2]}: {erro}")
            continue
        for chave in chaves:
            if resposta.get(chave) is not None:
                valores[chave] = resposta[chave]
        if "n_retas" in resposta:
            valores["n_retas"] = resposta["n_retas"]
    return valores, avisos


def ja_feitas(saida: Path) -> set[str]:
    if not saida.exists():
        return set()
    with saida.open(encoding="utf-8", newline="") as arquivo:
        return {linha["arquivo"] for linha in csv.DictReader(arquivo)}


def conjunto_raise() -> list[Path]:
    # Os `__recorte` sao outra condicao (recorte central sem reamostragem) e nao
    # entram: o protocolo da T04 e 256x256 reamostrado, como o corpus.
    return sorted(p for p in RAISE_DIR.glob("*.png")
                  if "__recorte" not in p.name)


def conjunto_coco(quantidade: int) -> list[Path]:
    """As **mesmas** 999 do controle de 07/09 — mesma funcao, mesma semente.

    Importar de `fpr_raise1k_todas` em vez de reimplementar e deliberado: se o
    sorteio mudar la, esta extracao acompanha, em vez de pontuar um conjunto
    silenciosamente diferente do que sera medido.
    """
    from automacao.fpr_raise1k_todas import imagens_coco
    return imagens_coco(quantidade)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--conjunto", choices=("raise1k", "coco"), required=True)
    parser.add_argument("--saida", type=Path, default=None)
    parser.add_argument("--limite", type=int, default=None)
    parser.add_argument("--timeout", type=int, default=300)
    args = parser.parse_args()
    if args.saida is None:
        args.saida = SAIDAS[args.conjunto]

    if args.conjunto == "raise1k":
        imagens = conjunto_raise()
        rotulo = 0
    else:
        imagens = conjunto_coco(len(conjunto_raise()))
        rotulo = 0
    if not imagens:
        print(f"ERRO: nenhuma imagem no conjunto {args.conjunto}", flush=True)
        return 1

    feitas = ja_feitas(args.saida)
    pendentes = [p for p in imagens if p.stem not in feitas]
    if args.limite:
        pendentes = pendentes[:args.limite]

    print(f"{args.conjunto}: {len(imagens)} imagens · ja feitas: {len(feitas)} · "
          f"a extrair: {len(pendentes)}", flush=True)
    if not pendentes:
        print("nada a fazer", flush=True)
        return 0

    args.saida.parent.mkdir(parents=True, exist_ok=True)
    novo = not args.saida.exists()
    inicio = time.perf_counter()
    falhas = acima_do_limite = 0

    with args.saida.open("a", newline="", encoding="utf-8") as arquivo:
        escritor = csv.DictWriter(arquivo, fieldnames=CAMPOS, extrasaction="ignore")
        if novo:
            escritor.writeheader()

        for indice, caminho in enumerate(pendentes, start=1):
            valores, avisos = escorear(caminho, args.timeout)
            if avisos:
                falhas += 1
                print(f"  ⚠️ {caminho.stem}: {' · '.join(avisos)}", flush=True)
            retas = valores.get("n_retas")
            if isinstance(retas, (int, float)) and retas > LIMITE_RETAS:
                acima_do_limite += 1

            escritor.writerow({
                "arquivo": caminho.stem, "split": "test", "label": rotulo,
                **valores,
            })
            arquivo.flush()  # nao perder a rodada inteira se o processo morrer

            if indice % 25 == 0 or indice == len(pendentes):
                decorrido = time.perf_counter() - inicio
                resta = decorrido / indice * (len(pendentes) - indice)
                print(f"  {indice}/{len(pendentes)} · {decorrido/60:.1f} min · "
                      f"faltam ~{resta/60:.1f} min", flush=True)

    print(f"\nconcluido · {falhas} imagens com falha de servico", flush=True)
    if acima_do_limite:
        # Nao invalida a rodada, mas precisa ser dito: acima de 250 retas o
        # escore passa a depender da amostragem do PointNet.
        print(f"⚠️ {acima_do_limite} imagens acima de {LIMITE_RETAS} retas — "
              f"nessas o escore de line_segment depende da amostragem",
              flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
