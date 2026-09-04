"""Compara a T01 deste projeto com uma reimplementacao independente.

Motivacao
---------
Os numeros da T01 no Capitulo 4 foram medidos com uma unica implementacao. Isso
deixa em aberto se eles sao propriedade da **tecnica** de Nataraj et al. (2019)
ou desta **implementacao** em particular -- o artigo e omisso em varios pontos
que o codigo obriga a decidir (deslocamento da coocorrencia, normalizacao,
posicao das ativacoes, regularizacao).

Uma segunda implementacao, escrita a partir do artigo sem ver a primeira,
responde a pergunta. Este script confronta as duas.

O que ele mede, em tres niveis
------------------------------
1. **Agregado**: AUC, acuracia, F1, FPR e FNR lado a lado.
2. **Significancia**: teste de DeLong sobre as duas AUCs. As duas implementacoes
   sao avaliadas no *mesmo* conjunto de teste, entao as AUCs sao correlacionadas
   e um teste nao pareado superestimaria a incerteza. E o mesmo procedimento que
   a Secao 4.3 usa para T05 contra T01, o que mantem os dois resultados
   reportaveis na mesma linguagem.
3. **Por imagem**: correlacao dos escores, taxa de concordancia no limiar de
   0,5, e -- o que interessa de verdade -- **quais** imagens as duas discordam,
   desagregado por gerador.

O terceiro nivel e o que distingue "as duas chegam ao mesmo numero" de "as duas
acertam as mesmas imagens". Duas implementacoes podem empatar na AUC decidindo
diferente em milhares de imagens, e isso e um achado, nao um empate.

A comparacao so e interpretavel se os dois lados forem **treinados e avaliados
sobre as mesmas particoes**. Os corpora ``corvi2024_30k`` e ``corvi2024_escala``
foram particionados de forma independente e nao sao aninhados: 18,2% do teste da
escala cai no treino do 30k, e 62,7% do teste do 30k cai no treino da escala.
Cruzar os dois vaza. Este script recusa a comparacao quando os dois lados nao
cobrem exatamente o mesmo conjunto de imagens.

Uso::

    # dois lados como CSV, cada um no formato da Secao 4 do contrato
    python automacao/comparar_implementacoes_t01.py \\
        --escores-original saidas/v1_escala/escores.csv \\
        --escores-novos    saidas/novo_escala/escores.csv \\
        --saida resultados/comparacao_implementacoes_t01

    # ou o lado original vindo do cache de escores ja medido no corpus de 30k
    python automacao/comparar_implementacoes_t01.py \\
        --protocolo standard \\
        --escores-novos saidas/novo_30k/escores.csv
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from codigo.configuracao import DATA_DIR, RESULTS_DIR  # noqa: E402
from codigo.metricas import (  # noqa: E402
    bootstrap_auc_ci,
    compute_metrics,
    delong_test,
)

# Cada protocolo tem um manifesto e um arquivo de escores em cache. O cache
# guarda os escores por imagem da implementacao original, na ordem da particao
# de teste do manifesto correspondente.
PROTOCOLOS = {
    "standard": ("manifesto30k_standard.csv", "standard__T01__T01__clean.npy"),
    "ood": ("manifesto30k_ood.csv", "ood__T01__T01__clean.npy"),
}

LIMIAR = 0.5


class ComparacaoError(RuntimeError):
    """Falha que invalida a comparacao. Sempre alta, nunca silenciosa."""


# ---------------------------------------------------------------------------
# Carga
# ---------------------------------------------------------------------------


def carregar_original(protocolo: str) -> pd.DataFrame:
    """Escores por imagem da implementacao original, com caminho e rotulo.

    O arquivo ``.npy`` e um vetor sem identificacao: sua i-esima posicao
    corresponde a i-esima linha da particao de teste do manifesto. A juncao com
    a outra implementacao e feita por caminho, entao a correspondencia posicional
    e reconstruida aqui, uma unica vez, e conferida pelo tamanho.
    """
    nome_manifesto, nome_escores = PROTOCOLOS[protocolo]

    manifesto = DATA_DIR / nome_manifesto
    escores = RESULTS_DIR / "scores" / nome_escores
    for caminho in (manifesto, escores):
        if not caminho.exists():
            raise ComparacaoError(f"arquivo necessario nao encontrado: {caminho}")

    teste = pd.read_csv(manifesto).query("split == 'test'").reset_index(drop=True)
    valores = np.load(escores)

    if len(teste) != len(valores):
        raise ComparacaoError(
            f"{nome_escores} tem {len(valores)} escores, mas a particao de teste "
            f"de {nome_manifesto} tem {len(teste)} linhas. A correspondencia "
            "posicional nao vale; a comparacao seria entre imagens diferentes."
        )

    return pd.DataFrame({
        "chave": teste["path"].map(normalizar_caminho),
        "label": teste["label"].astype(int),
        "generator": teste["generator"],
        "escore_original": valores.astype(np.float64),
    })


def carregar_novos(caminho: Path) -> pd.DataFrame:
    """Escores da reimplementacao, no formato da Secao 4 do contrato."""
    if not caminho.exists():
        raise ComparacaoError(f"escores da reimplementacao nao encontrados: {caminho}")

    frame = pd.read_csv(caminho)
    faltando = {"path", "label", "escore"} - set(frame.columns)
    if faltando:
        raise ComparacaoError(
            f"{caminho.name} nao segue o contrato: faltam as colunas {sorted(faltando)}"
        )

    fora = frame[(frame["escore"] < 0.0) | (frame["escore"] > 1.0)]
    if not fora.empty:
        raise ComparacaoError(
            f"{len(fora)} escores fora de [0, 1] em {caminho.name}; o contrato "
            "exige probabilidade, nao logito"
        )

    return pd.DataFrame({
        "chave": frame["path"].map(normalizar_caminho),
        "label_novo": frame["label"].astype(int),
        "escore_novo": frame["escore"].astype(np.float64),
    })


def normalizar_caminho(valor: str) -> str:
    """Chave de juncao estavel entre manifestos absolutos e relativos.

    O manifesto de 30k guarda caminho absoluto e o de escala guarda relativo;
    as duas implementacoes podem gravar qualquer um dos dois. O nome do arquivo
    precedido da pasta do gerador identifica a imagem sem ambiguidade nos dois
    corpora, e sobrevive a diferenca de separador entre plataformas.
    """
    partes = str(valor).replace("\\", "/").rstrip("/").split("/")
    return "/".join(partes[-2:]).lower()


def juntar(original: pd.DataFrame, novos: pd.DataFrame) -> pd.DataFrame:
    """Junta por imagem, recusando qualquer perda de linha.

    Uma juncao parcial produziria metricas sobre subconjuntos diferentes, que
    e o modo silencioso de errar: os numeros sairiam, plausiveis e errados.
    """
    if original["chave"].duplicated().any():
        raise ComparacaoError("chaves duplicadas nos escores originais")
    if novos["chave"].duplicated().any():
        raise ComparacaoError("chaves duplicadas nos escores da reimplementacao")

    juntos = original.merge(novos, on="chave", how="inner", validate="one_to_one")

    if len(juntos) != len(original):
        ausentes = len(original) - len(juntos)
        raise ComparacaoError(
            f"{ausentes} das {len(original)} imagens de teste nao aparecem nos "
            "escores da reimplementacao. As duas precisam ser avaliadas sobre o "
            "mesmo conjunto, ou a comparacao nao tem sentido."
        )

    divergentes = juntos[juntos["label"] != juntos["label_novo"]]
    if not divergentes.empty:
        raise ComparacaoError(
            f"{len(divergentes)} imagens com rotulo diferente entre as duas "
            "fontes. Indica manifesto ou particao trocada."
        )

    return juntos.drop(columns=["label_novo"])


# ---------------------------------------------------------------------------
# Analise
# ---------------------------------------------------------------------------


def comparar(juntos: pd.DataFrame, n_bootstrap: int) -> dict:
    """Os tres niveis: agregado, significancia e concordancia por imagem."""
    y = juntos["label"].to_numpy()
    a = juntos["escore_original"].to_numpy()
    b = juntos["escore_novo"].to_numpy()

    metricas = {
        "original": compute_metrics(y, a, LIMIAR),
        "reimplementacao": compute_metrics(y, b, LIMIAR),
    }

    ic = {
        "original": bootstrap_auc_ci(y, a, n_resamples=n_bootstrap),
        "reimplementacao": bootstrap_auc_ci(y, b, n_resamples=n_bootstrap),
    }

    # DeLong: as duas AUCs vem do mesmo conjunto de teste, logo sao
    # correlacionadas. A ordem (original, reimplementacao) faz a diferenca
    # positiva significar "a original tem AUC maior".
    delong = delong_test(y, a, b)

    predicao_a = (a >= LIMIAR).astype(int)
    predicao_b = (b >= LIMIAR).astype(int)
    concorda = predicao_a == predicao_b

    return {
        "n": int(len(juntos)),
        "metricas": metricas,
        "ic_auc_95": {k: list(v) for k, v in ic.items()},
        "delong": delong,
        "concordancia": {
            "taxa": float(concorda.mean()),
            "n_discordantes": int((~concorda).sum()),
            "pearson": float(np.corrcoef(a, b)[0, 1]),
            "spearman": float(pd.Series(a).corr(pd.Series(b), method="spearman")),
            # Onde uma acerta e a outra erra: e o que a AUC agregada esconde.
            "so_original_acerta": int(((predicao_a == y) & (predicao_b != y)).sum()),
            "so_reimplementacao_acerta": int(((predicao_b == y) & (predicao_a != y)).sum()),
        },
    }


def discordancia_por_gerador(juntos: pd.DataFrame) -> pd.DataFrame:
    """Desagrega a discordancia por gerador, ordenada pela taxa."""
    predicao_a = (juntos["escore_original"] >= LIMIAR).astype(int)
    predicao_b = (juntos["escore_novo"] >= LIMIAR).astype(int)

    tabela = juntos.assign(
        discorda=(predicao_a != predicao_b).astype(int),
        so_original=((predicao_a == juntos["label"]) & (predicao_b != juntos["label"])).astype(int),
        so_nova=((predicao_b == juntos["label"]) & (predicao_a != juntos["label"])).astype(int),
        delta=(juntos["escore_novo"] - juntos["escore_original"]).abs(),
    )

    resumo = tabela.groupby("generator", dropna=False).agg(
        n=("discorda", "size"),
        n_discordantes=("discorda", "sum"),
        so_original_acerta=("so_original", "sum"),
        so_nova_acerta=("so_nova", "sum"),
        delta_medio=("delta", "mean"),
    ).reset_index()

    resumo["taxa_discordancia"] = resumo["n_discordantes"] / resumo["n"]
    return resumo.sort_values("taxa_discordancia", ascending=False)


# ---------------------------------------------------------------------------
# Relato
# ---------------------------------------------------------------------------


def imprimir(resultado: dict, por_gerador: pd.DataFrame) -> None:
    m = resultado["metricas"]
    ic = resultado["ic_auc_95"]
    c = resultado["concordancia"]
    d = resultado["delong"]

    print(f"\n{'=' * 68}")
    print(f"Comparacao sobre {resultado['n']} imagens de teste")
    print("=" * 68)

    print(f"\n{'':<22}{'original':>14}{'reimplementacao':>18}")
    for rotulo, chave in (
        ("AUC", "auc"), ("Acuracia", "accuracy"), ("F1", "f1"),
        ("FPR", "fpr"), ("FNR", "fnr"),
    ):
        print(f"  {rotulo:<20}{m['original'][chave]:>14.4f}"
              f"{m['reimplementacao'][chave]:>18.4f}")

    print(f"\n  IC 95% da AUC")
    for nome in ("original", "reimplementacao"):
        print(f"    {nome:<18}[{ic[nome][0]:.4f}; {ic[nome][1]:.4f}]")

    print(f"\n  Teste de DeLong (AUCs correlacionadas, mesmo conjunto)")
    print(f"    diferenca         {d['difference']:+.4f}  (original menos reimplementacao)")
    print(f"    z                 {d['z']:+.4f}")
    print(f"    valor-p           {d['p_value']:.4f}")
    veredito = ("as duas AUCs diferem alem da variabilidade amostral"
                if d["p_value"] < 0.05 else
                "a diferenca de AUC nao excede a variabilidade amostral")
    print(f"    -> {veredito}")

    print(f"\n  Concordancia por imagem (limiar {LIMIAR})")
    print(f"    taxa              {c['taxa']:.4f}  ({c['n_discordantes']} discordantes)")
    print(f"    Pearson           {c['pearson']:.4f}")
    print(f"    Spearman          {c['spearman']:.4f}")
    print(f"    so a original acerta        {c['so_original_acerta']}")
    print(f"    so a reimplementacao acerta {c['so_reimplementacao_acerta']}")

    print(f"\n  Discordancia por gerador")
    print(por_gerador.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print()


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compara a T01 original com uma reimplementacao independente"
    )
    parser.add_argument("--escores-novos", type=Path, required=True,
                        help="escores.csv produzido pela reimplementacao")
    parser.add_argument("--protocolo", choices=sorted(PROTOCOLOS), default="standard")
    parser.add_argument("--saida", type=Path,
                        default=RESULTS_DIR / "comparacao_implementacoes_t01")
    parser.add_argument("--bootstrap", type=int, default=1000,
                        help="reamostragens para o IC da AUC")
    args = parser.parse_args()

    try:
        original = carregar_original(args.protocolo)
        novos = carregar_novos(args.escores_novos)
        juntos = juntar(original, novos)
    except ComparacaoError as erro:
        print(f"ERRO: {erro}")
        return 1

    resultado = comparar(juntos, args.bootstrap)
    por_gerador = discordancia_por_gerador(juntos)
    imprimir(resultado, por_gerador)

    args.saida.mkdir(parents=True, exist_ok=True)
    resultado["protocolo"] = args.protocolo
    resultado["escores_novos"] = str(args.escores_novos)

    destino_json = args.saida / f"comparacao_{args.protocolo}.json"
    destino_csv = args.saida / f"discordancia_por_gerador_{args.protocolo}.csv"
    destino_pares = args.saida / f"escores_pareados_{args.protocolo}.csv"

    destino_json.write_text(json.dumps(resultado, indent=2), encoding="utf-8")
    por_gerador.to_csv(destino_csv, index=False)
    juntos.to_csv(destino_pares, index=False)

    print(f"gravado em {args.saida}")
    print(f"  {destino_json.name}")
    print(f"  {destino_csv.name}")
    print(f"  {destino_pares.name}   (escores pareados, para inspecionar caso a caso)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
