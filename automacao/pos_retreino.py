"""O que fazer com o resultado do retreino das 22h de 05/09, sem ninguem acordado.

Roda depois que `noite_retreino.py` da [[Replicacao da T01]] termina (~02h15) e
deixa em disco um relatorio pronto para ler de manha. Duas partes independentes:
se uma falhar, a outra ainda entrega.

Parte 1 -- ler o retreino e dizer se a inversao aconteceu
---------------------------------------------------------
Le `saidas/retreino/resumo.csv` do repositorio da replicacao e reporta o numero
que decide: a **AUC do braco `A_balanceado` nos 4 geradores reservados**, com o
intervalo e com as duas sementes de repeticao como barra de erro.

O limiar de ~0,88 vem do Painel do cofre: acima dele a T01 retreinada supera a
T05 (0,8228) no protocolo OOD rigoroso, e a comparacao da A3 precisa ser
reaberta antes de escrever o Capitulo 4.

⚠️ **Este script nao conclui nada sozinho sobre significancia.** Ele compara
numeros de protocolos que nao sao o mesmo -- a T01 da replicacao nao e a T01 do
v1, e `RESULTADO_OOD.md` ja registra que so o *sentido* da falha e comparavel
entre as duas implementacoes. O que ele entrega e a leitura crua, sinalizada.

Parte 2 -- teto do que a fusao condicionada a degradacao pode render
-------------------------------------------------------------------
Em 05/09 mediu-se que a fusao **perde para a T01 sozinha** no protocolo por
familias (DeLong p < 1e-9). A fusao condicionada e o unico caminho restante que
poderia devolver a T05 uma vantagem de desempenho defensavel, porque ataca o
buraco conhecido: sob JPEG q50 a T05 (0,8162) fica **abaixo da T02 sozinha**
(0,8951), e o mecanismo esta entendido -- T01 e T03 sao descritores de alta
frequencia e a compressao destroi o sinal por construcao.

Antes de construir o estimador de condicao, mede-se o **teto**: uma fusao por
condicao, escolhida por um oraculo que sabe a condicao verdadeira. Se nem o
oraculo alcanca a T02 nas linhas de JPEG, o estimador nao vale a pena e a ideia
morre barata. E uma porta de decisao, nao uma proposta.

Tudo roda sobre escores em cache. Nenhuma imagem e processada, nenhum peso de
`pesos/` e tocado, nenhum `.npy` do cache e reescrito.

Uso::

    python automacao/pos_retreino.py
    python automacao/pos_retreino.py --sem-cofre   # nao escreve no Obsidian
"""

from __future__ import annotations

import argparse
import sys
import traceback
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ))

from codigo.tecnicas.t05_fusao.tecnica import T05Fusion, build_score_matrix  # noqa: E402

REPLICACAO = RAIZ.parent / "tcc3-t01-replicacao"
RESUMO_RETREINO = REPLICACAO / "saidas" / "retreino" / "resumo.csv"
LOG_RETREINO = REPLICACAO / "logs" / "retreino.log"
COFRE = Path.home() / "OneDrive" / "Documentos" / "TCC 3"
CACHE = RAIZ / "resultados" / "scores"
MANIFESTO_PADRAO = RAIZ / "data" / "manifesto30k_standard.csv"
SAIDA = RAIZ / "resultados" / "pos_retreino"

FONTES = ("T01", "T02", "T03")
LIMIAR_INVERSAO = 0.88          # Painel do cofre; AUC da T05 no OOD rigoroso e 0,8228
AUC_T05_OOD_RIGOROSO = 0.8228

CONDICOES = ("clean", "jpeg_q85", "jpeg_q70", "jpeg_q50",
             "noise_sigma1", "noise_sigma3", "noise_sigma5",
             "resize_85", "resize_70", "resize_50")

# Secao 5 de documentacao/RESULTADOS.md -- a guarda de alinhamento da parte 2.
AUC_PUBLICADA = {
    "clean":        {"T01": 0.9954, "T02": 0.9982, "T03": 0.8195},
    "jpeg_q85":     {"T01": 0.7530, "T02": 0.9811, "T03": 0.6771},
    "jpeg_q70":     {"T01": 0.7095, "T02": 0.9451, "T03": 0.6089},
    "jpeg_q50":     {"T01": 0.5842, "T02": 0.8951, "T03": 0.4415},
    "noise_sigma1": {"T01": 0.9015, "T02": 0.9982, "T03": 0.8198},
    "noise_sigma3": {"T01": 0.8123, "T02": 0.9951, "T03": 0.7900},
    "noise_sigma5": {"T01": 0.6682, "T02": 0.9904, "T03": 0.7713},
    "resize_85":    {"T01": 0.9748, "T02": 0.9914, "T03": 0.8003},
    "resize_70":    {"T01": 0.9508, "T02": 0.9798, "T03": 0.7644},
    "resize_50":    {"T01": 0.8446, "T02": 0.8690, "T03": 0.6037},
}


def tabela_md(d: pd.DataFrame, casas: int = 4) -> str:
    """Tabela markdown sem depender de `tabulate`, que nao esta no ambiente."""
    def formatar(v) -> str:
        return f"{v:.{casas}f}" if isinstance(v, (float, np.floating)) else str(v)

    cabecalho = "| " + " | ".join(d.columns) + " |"
    regua = "|" + "|".join("---" for _ in d.columns) + "|"
    corpo = ["| " + " | ".join(formatar(v) for v in linha) + " |"
             for linha in d.itertuples(index=False)]
    return "\n".join([cabecalho, regua, *corpo])


# ----------------------------------------------------------------------
# Parte 1 -- o retreino
# ----------------------------------------------------------------------

def parte1() -> list[str]:
    linhas = ["## Parte 1 — o retreino das 22h", ""]

    if not RESUMO_RETREINO.exists():
        linhas += [
            "⛔ **`saidas/retreino/resumo.csv` não existe.** O retreino não produziu "
            "resultado — nem parcial. `retreino_noturno.py` grava o resumo a cada braço "
            "concluído, então a ausência do arquivo significa que nenhum dos quatro "
            "terminou.", "",
            f"Log esperado em `{LOG_RETREINO}` — "
            + ("existe, ver as últimas linhas abaixo." if LOG_RETREINO.exists()
               else "**também não existe**, o que aponta para a tarefa não ter disparado."),
            "",
        ]
        if LOG_RETREINO.exists():
            cauda = LOG_RETREINO.read_text(encoding="utf-8", errors="replace").splitlines()[-25:]
            linhas += ["```"] + cauda + ["```", ""]
        return linhas

    d = pd.read_csv(RESUMO_RETREINO)
    linhas += [f"`{RESUMO_RETREINO.relative_to(REPLICACAO.parent)}` — "
               f"**{len(d)} de 4 braços** concluíram.", ""]
    if len(d) < 4:
        linhas += ["⚠️ Braços faltando: **"
                   + ", ".join(sorted({"A_balanceado", "B_volume", "A_semente7",
                                       "A_semente123"} - set(d["braco"]))) + "**. "
                   "Um braço que falha não derruba os outros, mas as barras de erro "
                   "dependem das duas sementes.", ""]

    linhas += [tabela_md(d), ""]

    principal = d[d["braco"] == "A_balanceado"]
    if principal.empty:
        linhas += ["⛔ **O braço `A_balanceado` não concluiu** — é ele que carrega o "
                   "número que decide. Sem ele, nada a concluir sobre a inversão.", ""]
        return linhas

    auc = float(principal["auc_reservados"].iloc[0])
    baixo = float(principal["ic_baixo"].iloc[0])
    alto = float(principal["ic_alto"].iloc[0])
    sementes = d[d["braco"].str.startswith("A_semente")]["auc_reservados"].to_numpy()

    linhas += ["### O número que decide", "",
               f"**`A_balanceado` nos 4 reservados: AUC {auc:.4f}**, IC 95% "
               f"[{baixo:.4f}; {alto:.4f}]."]
    if len(sementes):
        todas = np.r_[auc, sementes]
        linhas += [f"Com as {len(sementes)} sementes de repetição: média "
                   f"**{todas.mean():.4f}**, amplitude {todas.min():.4f}–{todas.max():.4f} "
                   f"(desvio {todas.std(ddof=1):.4f})." if len(todas) > 1 else ""]
    else:
        linhas += ["⚠️ Nenhum braço de semente concluiu — **sem barra de erro de "
                   "inicialização**, que é justamente o que faltava para defender a "
                   "diferença numa banca."]
    linhas += [""]

    if auc > LIMIAR_INVERSAO:
        veredito = (
            f"🔴 **Acima do limiar de {LIMIAR_INVERSAO:.2f}.** A T01 retreinada "
            f"({auc:.4f}) fica acima da T05 no OOD rigoroso ({AUC_T05_OOD_RIGOROSO:.4f}). "
            "Somado ao que já se mediu no protocolo por famílias (T01 vence, "
            "p < 1e-9), **a manchete de AUC da T05 caiu nos dois protocolos**."
        )
    elif baixo > AUC_T05_OOD_RIGOROSO:
        veredito = (
            f"🟠 **Abaixo de {LIMIAR_INVERSAO:.2f}, mas o IC inteiro está acima da T05** "
            f"({AUC_T05_OOD_RIGOROSO:.4f}). A vantagem da fusão não sobrevive como "
            "afirmação, ainda que a margem seja menor do que o limiar previa."
        )
    else:
        veredito = (
            f"🟢 **{auc:.4f}, abaixo do limiar de {LIMIAR_INVERSAO:.2f}.** No protocolo "
            f"rigoroso a T05 ({AUC_T05_OOD_RIGOROSO:.4f}) segue à frente da T01 "
            "retreinada. ⚠️ Isso **não** reabilita a manchete de AUC: no protocolo por "
            "famílias a T01 já vence de forma significativa. O resultado passa a ser "
            "*a vantagem depende do protocolo*, que é mais fraco e precisa ser dito."
        )
    linhas += ["### Leitura", "", veredito, "",
               "⚠️ **Comparação entre implementações diferentes.** A T01 da replicação "
               "não é a T01 do v1; `RESULTADO_OOD.md` registra que só o *sentido* da "
               "falha é comparável. O número acima orienta a decisão, não fecha a A3 "
               "sozinho.", ""]

    for coluna, gerador in (("auc_sd3", "Stable Diffusion 3"), ("auc_dalle3", "DALL·E 3"),
                            ("auc_firefly", "Adobe Firefly"), ("auc_mjv61", "Midjourney v6.1")):
        if coluna in principal:
            linhas += [f"- {gerador}: **{float(principal[coluna].iloc[0]):.4f}**"]
    linhas += [""]
    return linhas


# ----------------------------------------------------------------------
# Parte 2 -- teto da fusao condicionada
# ----------------------------------------------------------------------

def _rotulos_robustez() -> np.ndarray:
    """Reconstroi a amostra de 1.000 imagens do protocolo de robustez.

    `runner.run_robustness` sorteia com `default_rng(42)` estratificando por
    classe, na ordem (reais, sinteticas). A reconstrucao e deterministica, e a
    guarda de AUC abaixo confirma que ela esta certa antes de qualquer uso.
    """
    d = pd.read_csv(MANIFESTO_PADRAO)
    y = d[d.split == "test"].reset_index(drop=True).label.to_numpy()
    rng = np.random.default_rng(42)
    escolhidos = np.concatenate([
        rng.choice(np.flatnonzero(y == rotulo),
                   size=min(1000 // 2, int((y == rotulo).sum())), replace=False)
        for rotulo in (0, 1)
    ])
    return y[escolhidos]


def parte2() -> list[str]:
    linhas = ["## Parte 2 — teto da fusão condicionada à degradação", ""]

    y = _rotulos_robustez()
    escores = {c: {t: np.load(CACHE / f"robustness__{t}__{t}__{c}.npy") for t in FONTES}
               for c in CONDICOES}

    divergencias = []
    for c in CONDICOES:
        for t in FONTES:
            medido = roc_auc_score(y, escores[c][t])
            alvo = AUC_PUBLICADA[c][t]
            if abs(medido - alvo) > 5e-4:
                divergencias.append(f"{c}/{t}: medido {medido:.4f}, publicado {alvo:.4f}")
    if divergencias:
        linhas += ["⛔ **Guarda de alinhamento reprovada** — os rótulos reconstruídos não "
                   "reproduzem a tabela da Seção 5 de `RESULTADOS.md`. Nada foi ajustado.",
                   "", "```"] + divergencias[:10] + ["```", ""]
        return linhas
    linhas += ["Alinhamento conferido: as 30 AUCs da Seção 5 de `RESULTADOS.md` foram "
               "reproduzidas a partir do cache antes de qualquer ajuste.", ""]

    # Metade das imagens ajusta, metade avalia -- por imagem, nao por condicao,
    # para que nenhuma imagem apareca nos dois lados em condicao nenhuma.
    rng = np.random.default_rng(42)
    ordem = rng.permutation(len(y))
    ajuste, avaliacao = ordem[: len(y) // 2], ordem[len(y) // 2:]

    def matriz(c: str, idx: np.ndarray) -> np.ndarray:
        return build_score_matrix({t: escores[c][t][idx] for t in FONTES}, len(idx))

    # (a) pratica atual: uma fusao ajustada so na condicao limpa.
    so_limpa = T05Fusion(seed=42).fit_scores(matriz("clean", ajuste), y[ajuste])
    # (b) uma fusao ajustada com todas as condicoes misturadas, sem saber qual e.
    empilhada = np.vstack([matriz(c, ajuste) for c in CONDICOES])
    agrupada = T05Fusion(seed=42).fit_scores(empilhada, np.tile(y[ajuste], len(CONDICOES)))
    # (c) o teto: uma fusao por condicao, escolhida por um oraculo.
    por_condicao = {c: T05Fusion(seed=42).fit_scores(matriz(c, ajuste), y[ajuste])
                    for c in CONDICOES}

    registros = []
    for c in CONDICOES:
        m = matriz(c, avaliacao)
        ya = y[avaliacao]
        registros.append(dict(
            condicao=c,
            T02_sozinha=roc_auc_score(ya, escores[c]["T02"][avaliacao]),
            T05_so_limpa=roc_auc_score(ya, so_limpa.predict_proba_scores(m)),
            T05_agrupada=roc_auc_score(ya, agrupada.predict_proba_scores(m)),
            T05_oraculo=roc_auc_score(ya, por_condicao[c].predict_proba_scores(m)),
        ))
    df = pd.DataFrame(registros)
    SAIDA.mkdir(parents=True, exist_ok=True)
    df.to_csv(SAIDA / "fusao_condicionada.csv", index=False)

    linhas += ["AUC por condição, avaliada nas 500 imagens que não entraram em ajuste "
               "nenhum:", "", tabela_md(df), ""]

    jpeg = df[df.condicao.str.startswith("jpeg")]
    ganha_do_t02 = int((jpeg.T05_oraculo > jpeg.T02_sozinha).sum())
    ganho = float((df.T05_oraculo - df.T05_so_limpa).mean())
    linhas += ["### Porta de decisão", "",
               f"O oráculo supera a T02 sozinha em **{ganha_do_t02} das 3 condições de "
               f"JPEG**, e ganha em média **{ganho:+.4f}** de AUC sobre a prática atual "
               "(fusão ajustada só no limpo).", ""]
    if ganha_do_t02 == 3:
        linhas += ["🟢 **Vale construir o estimador de condição.** Mesmo com o oráculo "
                   "sendo teto inalcançável, há folga sobre a T02 justamente onde a "
                   "fusão hoje perde. O próximo passo é estimar qualidade JPEG do "
                   "próprio arquivo — é leitura de cabeçalho, não inferência.", ""]
    elif ganha_do_t02 > 0:
        linhas += ["🟠 **Parcial.** O oráculo só recupera parte das condições de JPEG. "
                   "Um estimador real fica abaixo disso — decidir se o ganho parcial "
                   "paga a complexidade de mais um componente no Capítulo 4.", ""]
    else:
        linhas += ["🔴 **Não vale construir.** Nem sabendo a condição de antemão a fusão "
                   "alcança a T02 sozinha sob JPEG. A ideia morre aqui, barata, e isso "
                   "é resultado negativo publicável: **quando uma fonte domina numa "
                   "condição, reponderar não substitui usar só ela.**", ""]
    return linhas


# ----------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sem-cofre", action="store_true",
                        help="nao escreve o relatorio no cofre Obsidian")
    args = parser.parse_args()

    # O Agendador roda sem console e o Windows usa cp1252 por padrao: sem isto
    # o `print` do relatorio quebra em qualquer acento ou emoji.
    for fluxo in (sys.stdout, sys.stderr):
        try:
            fluxo.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass

    agora = datetime.now()
    relatorio = [
        "---", "tipo: resultados", "tags: [tcc3, resultados]",
        f"atualizado: {agora:%Y-%m-%d}", "---", "",
        "# Relatório pós-retreino",
        "",
        f"> Gerado automaticamente por `automacao/pos_retreino.py` em "
        f"**{agora:%d/%m/%Y às %H:%M}**, sem ninguém acompanhando. "
        "As duas partes são independentes. Ver [[Plano pós-retreino]].",
        "",
    ]

    for nome, funcao in (("parte 1", parte1), ("parte 2", parte2)):
        try:
            relatorio += funcao()
        except Exception:                                   # noqa: BLE001
            relatorio += [f"## Falha na {nome}", "",
                          "⛔ A execução quebrou. Traceback abaixo — a outra parte "
                          "continuou.", "", "```"] + traceback.format_exc().splitlines() + ["```", ""]

    texto = "\n".join(relatorio)
    SAIDA.mkdir(parents=True, exist_ok=True)
    (SAIDA / "relatorio.md").write_text(texto, encoding="utf-8")
    if not args.sem_cofre and COFRE.exists():
        (COFRE / "Resultados" / "Relatório pós-retreino.md").write_text(texto, encoding="utf-8")
    print(texto)
    return 0


def _abrir_log():
    """Redireciona a saida para arquivo -- o Agendador nao da console.

    Mesmo motivo de `noite_retreino.py` da replicacao: `schtasks` executa sem
    terminal anexado e stdout vai para lugar nenhum, entao um traceback fora do
    `try` de cada parte desapareceria. `encoding` explicito porque o console do
    Windows usa cp1252 e este relatorio tem acentuacao e emoji.
    """
    caminho = RAIZ / "logs" / "pos_retreino.log"
    caminho.parent.mkdir(parents=True, exist_ok=True)
    arquivo = open(caminho, "a", encoding="utf-8", buffering=1)
    sys.stdout = sys.stderr = arquivo
    return arquivo


if __name__ == "__main__":
    if sys.stdout is None or not sys.stdout.isatty():
        _abrir_log()
    sys.exit(main())
