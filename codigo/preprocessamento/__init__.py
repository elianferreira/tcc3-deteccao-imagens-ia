"""Tudo que acontece com a imagem ANTES de ela chegar as tecnicas.

A ordem em que uma imagem atravessa este pacote:

    1. validacao_envio.py   o arquivo e aceitavel? (RN01 formato, RN02 10 MB)
                            o formato e conferido pelo CONTEUDO, nao pela
                            extensao -- o corpus tem um .png que e JPEG por
                            dentro (adobe_firefly).

    2. normalizacao.py      menor lado a 256 px com LANCZOS, recorte central,
                            RGB, gravado em PNG. E a condicao em que TODOS os
                            numeros do Capitulo 4 foram medidos.

    3. envio_interface.py   a politica de resolucao por tecnica na interface:
                            T01/T03/T04 recebem 256 px; a T02 recebe resolucao
                            nativa com teto de 1.536 px, porque e espectral.

    perturbacoes.py         o protocolo de robustez (secao 3.6.1): JPEG,
                            ruido gaussiano e reescala aplicados de proposito
                            para medir degradacao.

Por que este pacote existe separado
-----------------------------------
Ate 31/08/2026 a interface entregava o upload CRU as cinco tecnicas, enquanto
todos os numeros publicados foram medidos sobre imagens normalizadas. T01 e
T02 trocavam de lado conforme a resolucao do envio. O pre-processamento nao e
detalhe de implementacao: e o que decide se o que aparece na tela corresponde
ao que foi medido.
"""
