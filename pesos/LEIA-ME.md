# Os modelos de fusão da T05 — qual produziu qual número

> Escrito em 25/09/2026, depois da auditoria registrada no cofre em
> `Auditorias/Qual modelo produziu cada número da T05.md`. Existe porque a
> pergunta "de qual modelo saiu este número?" custou uma auditoria inteira, e
> não deve custar de novo.

Há **seis** arquivos `t05_fusion*.pkl` nesta pasta. Eles não são variações
cosméticas: no mesmo conjunto de teste, a FPR entre eles varia por um fator de
**9,5×**.

## A tabela que resolve a dúvida

FPR no limiar 0,5 sobre a partição `test` de `manifesto30k_standard.csv`
— **n = 9.000, das quais 4.500 reais**. É o conjunto da Seção 4.1.

| arquivo | fontes | coef. T04 | FPR | falsos positivos |
|---|---|---|---|---|
| `t05_fusion.pkl` | 3 | 0,0000 | **0,0400** | 180/4500 |
| `t05_fusion__variante_A.pkl` | 3 | 0,0000 | 0,002667 | 12/4500 |
| `t05_fusion__variante_B.pkl` | 3 | 0,0000 | 0,0400 | 180/4500 |
| `t05_fusion__quatro_fontes.pkl` | 4 | −0,2512 | 0,004444 | 20/4500 |
| **`t05_fusion__quatro_fontes__16-08_publicado.pkl`** | **4** | **−0,0403** | **0,004222** | **19/4500** |
| `t05_fusion__quatro_fontes_ood.pkl` | 4 | +0,2513 | 0,0400 | 180/4500 |

`t05_fusion.pkl` é byte a byte a `variante_B`.

## ⛔ O que precisa ser sabido antes de citar qualquer FPR da T05

**O número publicado na Seção 4.1 — FPR 0,42% — é o do modelo de QUATRO
fontes, na versão de 16/08.** Não é o de `t05_fusion.pkl`, que é o que o código
carrega por padrão (`TCC3_WEIGHTS_DIR`) e o que a interface usa: esse dá
**0,0400** no mesmo conjunto, 9,5× mais.

Os dois números estão corretos. São de modelos diferentes, e o texto da
dissertação não dizia isso.

## Por que existe um arquivo com `16-08_publicado` no nome

O modelo de quatro fontes foi **reajustado em 18/08** (commit `e006e8c`, *"Mede
T05 com T04 completa"*), quando as três representações da T04 passaram a entrar
no lugar de só o objeto-sombra. O reajuste sobrescreveu o arquivo.

Resultado: de 18/08 até 25/09 o repositório **não continha** o modelo que gerou
o número publicado. Ele só existia no histórico do git. Restaurado com:

    git show 83ae954:weights/t05_fusion__quatro_fontes.pkl

⚠️ **Nenhum arquivo foi sobrescrito para restaurá-lo** — o de 18/08 continua em
`t05_fusion__quatro_fontes.pkl`, com o nome que a Seção 4.8 cita.

## E o reajuste mudou mais do que o commit dizia

| fonte | 16/08 (publicado) | 18/08 (em disco) |
|---|---|---|
| T01 | +3,4061 | +3,4303 |
| T02 | +4,2525 | +4,2661 |
| T03 | +0,5063 | +0,5205 |
| **T04** | **−0,0403** | **−0,2512** |

A mensagem do commit afirma que *"as três representações não mudam a fusão"*.
T01, T02 e T03 de fato mal se movem — **o peso da T04 muda 6×**.

Isso importa porque a Seção 4.8 argumenta que *"a fusão aprendeu a ignorar T04:
peso praticamente nulo, duas ordens de grandeza abaixo de T01 e T02"*. Com
−0,0403 o argumento se sustenta; com **−0,2512** ele é uma ordem abaixo, e não
é desprezível.

## Somas de verificação

    edceef33c606fbd4   t05_fusion__quatro_fontes__16-08_publicado.pkl
    d581cf624d09e73b   t05_fusion__quatro_fontes.pkl
    921f092c58ac22a3   t05_fusion.pkl

(sha256, 16 primeiros dígitos. Conferir antes de atribuir um número a um
modelo.)
