"""Politica de resolucao por tecnica, aplicada a cada upload da interface.

Os dois gerenciadores de contexto daqui decidem QUAL imagem cada tecnica ve.
Ambos removem seus temporarios na saida, inclusive sob excecao (RNF03).

As constantes TCC3_T02_NATIVA e TETO_T02 vivem AQUI. Quem precisar
sobrescreve-las em teste deve remendar este modulo, nao a interface.
"""

from __future__ import annotations

import contextlib
import os
import tempfile
from collections.abc import Iterator
from pathlib import Path

from PIL import Image

from ..configuracao import IMAGE_SIZE
from .normalizacao import resize_and_center_crop


# ---------------------------------------------------------------------------
# Politica de resolucao por tecnica -- de 31/08/2026
# ---------------------------------------------------------------------------
#
# Cada tecnica recebe a imagem na condicao em que foi medida. T01, T03 e T04
# foram treinadas e avaliadas em 256x256; a T02 e espectral e foi verificada em
# resolucao nativa (secao 4.9: o controle do glide da 0,9626 contra 0,902
# publicado).
#
# Medido em adobe_firefly_00002 (2688x1536, sintetica):
#
#     tecnica    nativa    256 px
#     T01          2,5%     97,7%
#     T02         99,8%      0,0%
#     T04         69,8%     69,7%   (mesma resposta, e 12x mais cara em nativa)
#
# Nenhuma resolucao unica funciona: com tudo normalizado a T05 da 2,5%, com tudo
# nativo da 19,5% -- as duas erram. Com a T02 em nativa e o resto normalizado, a
# T05 da 99,3%.
#
# A T05 continua sendo alimentada pelo escore de 256 px, e nao pelo nativo. A
# fusao foi ajustada nessa distribuicao, e troca-la aqui mediria o reajuste em
# vez da mudanca (armadilha 6 de ESTADO_ATUAL.md). Por isso a T02 e avaliada
# duas vezes: o nativo vai para a tela, o de 256 px vai para a fusao.
#
# TETO: a T02 em 4,13 MPx consumiu 5.931 MiB dos 6.144 da placa e derrubou os
# servicos T04 do WSL2 por esgotamento de memoria. A RN02 aceita 10 MB, que
# podem ser bem maiores. Sem teto isto nao e operavel neste hardware.
TCC3_T02_NATIVA = os.environ.get("TCC3_T02_NATIVA", "1") != "0"
TETO_T02 = int(os.environ.get("TCC3_T02_TETO", "1536"))




@contextlib.contextmanager
def envio_normalizado(origem: Path) -> Iterator[Path]:
    """Poe o upload na mesma condicao em que as tecnicas foram medidas.

    Por que isto existe
    -------------------
    Todos os numeros do Capitulo 4 foram medidos sobre o corpus normalizado por
    ``automacao/normalize_corpus.py``: menor lado a 256 px com LANCZOS, recorte
    central, RGB. Ate 31/08/2026 a interface entregava o arquivo **cru** as
    cinco tecnicas, e um envio de 1024 px colocava T01, T03 e T04 fora da
    condicao em que foram treinadas e avaliadas.

    O efeito era visivel na tela. Com ``dalle3_00000`` (sintetica), o mesmo
    arquivo nas duas resolucoes:

        =======  ==============  ==================
        Tecnica     1024 px cru  256 px normaliz.
        =======  ==============  ==================
        T01         0,0% errado      99,5% correto
        T02       99,9% correto        0,0% errado
        =======  ==============  ==================

    T01 e T02 trocavam de lado exatamente. Nao e defeito de nenhuma das duas: a
    T01 foi treinada em 256 px e a T02 e espectral, medida em resolucao nativa
    por construcao. Normalizar acerta a T01 e piora a T02 na tela -- e a perda
    de 17,8 p.p. da secao 4.9, que e resultado medido, nao dano deste conserto.

    Efeito colateral util: uma imagem grande nunca mais chega crua aos modelos,
    o que elimina o caso de 214 s observado em 1024 px no processo frio.

    Custo
    -----
    Uma imagem ja conforme -- 256 x 256 em RGB, que e o caso do corpus inteiro
    -- e devolvida sem copia, para que ``automacao/medir_rnf01_com_t04.py`` siga
    medindo o mesmo caminho de antes. Sobra apenas a abertura do arquivo para
    conferir dimensoes e modo.

    RNF03: o arquivo normalizado vive num diretorio temporario removido na
    saida do contexto, mesmo em caso de excecao.
    """
    with Image.open(origem) as imagem:
        ja_conforme = imagem.size == (IMAGE_SIZE, IMAGE_SIZE) and imagem.mode == "RGB"
        # A conversao acontece dentro do ``with`` porque ``resize_and_center_crop``
        # le os pixels; o descritor precisa continuar aberto ate aqui.
        normalizada = None if ja_conforme else resize_and_center_crop(imagem, IMAGE_SIZE)
        if normalizada is not None:
            # ``crop`` e preguicoso no PIL. Forcar a leitura aqui garante que
            # nada dependa do descritor depois que o ``with`` o fechar.
            normalizada.load()

    if normalizada is None:
        # Fora do ``with``: o arquivo ja esta fechado quando as tecnicas o leem.
        # T02 abre a imagem num subprocesso e o Windows nao gosta de dois
        # descritores concorrentes sobre o mesmo arquivo.
        yield origem
        return

    with tempfile.TemporaryDirectory(prefix="tcc3_envio_") as pasta:
        # PNG: mesmo formato de saida da normalizacao do corpus, e sem
        # recompressao com perdas sobre o que sera analisado.
        destino = Path(pasta) / "envio_normalizado.png"
        normalizada.save(destino, format="PNG")
        normalizada.close()
        yield destino


@contextlib.contextmanager
def envio_para_t02(origem: Path) -> Iterator[Path | None]:
    """Entrega a T02 a maior resolucao que esta maquina aguenta.

    A T02 e literalmente *"Any-Resolution AI-Generated Image Detection by
    **Spectral** Learning"*: seu sinal e a distribuicao espectral, e reamostrar
    para 256x256 destroi a evidencia de alta frequencia que ela explora. A
    secao 4.9 mede isso -- media 17,8 p.p. abaixo do publicado --, e o controle
    do ``glide`` (nativo 256 px, intocado pela normalizacao) fecha a questao ao
    medir 0,9626 contra 0,902 publicado.

    Por que ha um teto
    ------------------
    Em 31/08/2026, a T02 sobre ``adobe_firefly_00002`` (2688x1536, 4,13 MPx)
    consumiu **5.931 MiB dos 6.144** da placa, com a interface parada. Com a
    interface no ar, a mesma chamada esgotou a memoria e **derrubou os dois
    servicos T04 do WSL2**. A RN02 aceita arquivos de 10 MB, que podem ser bem
    maiores que isso -- sem teto, um envio grande derruba a aplicacao.

    O teto vem da 4.9: nao ha dose-resposta, o efeito e de limiar -- reducao de
    2x fica dentro da tolerancia de 5 p.p., e so a partir de ~3,5x a degradacao
    satura. O padrao de 1.536 px no maior lado mantem a maioria dos envios
    dentro dessa faixa. **Nao e teto medido**; a medicao de ~200 imagens que o
    fixaria esta registrada como pergunta em aberto no ESTADO_ATUAL.md.

    Devolve ``None`` quando a politica esta desligada
    (``TCC3_T02_NATIVA=0``), caso em que a T02 volta a receber os 256 px como
    todas as demais.
    """
    if not TCC3_T02_NATIVA:
        yield None
        return

    with Image.open(origem) as imagem:
        largura, altura = imagem.size
        maior = max(largura, altura)
        if maior <= TETO_T02:
            # Ja cabe: entrega o proprio arquivo, sem reamostrar. Este e o caso
            # que preserva integralmente o sinal espectral da T02.
            reduzida = None
        else:
            escala = TETO_T02 / maior
            destino_tam = (max(1, round(largura * escala)), max(1, round(altura * escala)))
            reduzida = imagem.convert("RGB").resize(destino_tam, Image.LANCZOS)
            reduzida.load()

    if reduzida is None:
        yield origem
        return

    with tempfile.TemporaryDirectory(prefix="tcc3_t02_") as pasta:
        destino = Path(pasta) / "envio_t02.png"
        reduzida.save(destino, format="PNG")
        reduzida.close()
        yield destino
