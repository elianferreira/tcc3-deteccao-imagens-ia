"""Passo 1 - manifesto da replica da T01 para a recalibracao da T05 da v2.

Por que o protocolo e 256 px, e nao nativo
------------------------------------------
A ideia inicial era calibrar sob o protocolo de exibicao da v2 (T02 e T03 em
resolucao nativa). **Medido em 08/09/2026, isso nao e possivel com este
corpus**: em resolucao nativa ha dois confundidores perfeitos, sem uma unica
imagem de sobreposicao entre as classes.

    formato      reais 3.150 = 100% .jpg
                 sinteticas 11.638 = 100% .png/.webp

    resolucao    reais      menor lado 264 a 480
                 sinteticas menor lado 512 a 2.048

`min(largura, altura) < 512` sozinho e um detector de AUC 1,0. Uma fusao
ajustada ali aprenderia resolucao e formato, exibiria AUC quase perfeita e nao
mediria sintese nenhuma -- e a T02, que e *any-resolution* por construcao,
seria o veiculo principal, nao so a T03.

E exatamente o vies que o corpus normalizado existe para remover. Entao a
fusao e calibrada em **256 px nas quatro fontes**, como no v1, e a tela passa a
calcular dois vetores por envio: o de exibicao (resolucao por tecnica) e o de
256 px que alimenta a T05. A divergencia entre os dois e declarada na tela, nao
escondida.

O que este passo faz
--------------------
A unica fonte que falta em cache e a T01 da v2 -- a reimplementacao, que nunca
foi pontuada sobre o `corvi2024_30k_norm`. As outras tres ja existem em
`resultados/scores/ood__T0*__fusion.npy` (2.350) e nos CSV de componentes da
T04. Este script monta o manifesto que a replica consome.

Vazamento: a replica treinou no `corvi2024_escala`, e parte do 30k esta la.
A coluna `limpa` marca o que pode ser usado; o passo 3 filtra por ela.

O CLI da replica so aceita train/val/test, entao `fusion` e gravada como `val`.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

RAIZ = Path(__file__).resolve().parents[2]
NORM = RAIZ / "data" / "corvi2024_30k_norm"
SAIDA = RAIZ / "resultados" / "t05_v2"


def chave(valor: str) -> str:
    partes = str(valor).replace("\\", "/").rstrip("/").split("/")
    return "/".join(partes[-2:]).rsplit(".", 1)[0].lower()


def main() -> int:
    ood = pd.read_csv(RAIZ / "data" / "manifesto30k_ood.csv")
    escala = pd.read_csv(RAIZ / "data" / "manifesto_escala_standard.csv")
    ood["k"] = ood["path"].map(chave)
    escala["k"] = escala["path"].map(chave)
    treino_replica = set(escala.query("split == 'train'")["k"])

    alvo = ood[ood["split"].isin(["fusion", "test"])].copy()
    alvo["limpa"] = ~alvo["k"].isin(treino_replica)

    manifesto = pd.DataFrame({
        "path": alvo["path"].map(
            lambda p: str(Path(p).relative_to(NORM)).replace("\\", "/")
        ),
        "label": alvo["label"].astype(int),
        "generator": alvo["generator"],
        "source": alvo["generator"].map(lambda g: "coco" if g == "real" else ""),
        "split": alvo["split"].map({"fusion": "val", "test": "test"}),
        "width": 256,
        "height": 256,
    })

    SAIDA.mkdir(parents=True, exist_ok=True)
    manifesto.to_csv(SAIDA / "manifesto_replica.csv", index=False)
    # A ordem e preservada: o passo 3 junta por caminho, nao por posicao, mas a
    # etiqueta de limpeza precisa acompanhar as mesmas linhas.
    alvo[["path", "k", "split", "label", "generator", "limpa"]].to_csv(
        SAIDA / "indice_limpeza.csv", index=False
    )

    print("gravado em", SAIDA)
    print(f"   manifesto_replica.csv  {len(manifesto)} linhas")
    resumo = alvo.groupby(["split", "limpa"]).size().unstack(fill_value=0)
    print("\nlinhas por particao (limpa = fora do treino da replica):")
    print(resumo.to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
