"""Sonda a viabilidade de ler o LSUN por requisicoes parciais.

Os arquivos do LSUN objects somam mais de 1 TB, mas sao precisas apenas ~4.500
imagens por categoria. O servidor aceita Range (206), entao e possivel expor o
arquivo remoto como objeto tipo-arquivo e deixar o modulo zipfile ler apenas o
diretorio central e as entradas desejadas.

Esta sonda responde tres perguntas:
1. O zipfile consegue abrir o arquivo remoto?
2. Qual a estrutura de nomes dentro do zip?
3. Ela permite casar com os nomes da lista oficial (bus_02020.png)?
"""

import io
import time
import urllib.request
import zipfile

URL = "http://dl.yf.io/lsun/objects/bus.zip"


class ArquivoRemoto(io.RawIOBase):
    """Expoe uma URL como arquivo pesquisavel, via requisicoes Range."""

    def __init__(self, url: str):
        self.url = url
        self.posicao = 0
        self.requisicoes = 0
        self.bytes_lidos = 0
        pedido = urllib.request.Request(url, method="HEAD")
        with urllib.request.urlopen(pedido, timeout=60) as resposta:
            self.tamanho = int(resposta.headers["Content-Length"])

    def seekable(self) -> bool:
        return True

    def readable(self) -> bool:
        return True

    def seek(self, deslocamento, de_onde=0):
        if de_onde == 0:
            self.posicao = deslocamento
        elif de_onde == 1:
            self.posicao += deslocamento
        else:
            self.posicao = self.tamanho + deslocamento
        return self.posicao

    def tell(self) -> int:
        return self.posicao

    def read(self, quantidade=-1) -> bytes:
        if quantidade is None or quantidade < 0:
            quantidade = self.tamanho - self.posicao
        if quantidade == 0 or self.posicao >= self.tamanho:
            return b""
        fim = min(self.posicao + quantidade, self.tamanho) - 1
        pedido = urllib.request.Request(
            self.url, headers={"Range": f"bytes={self.posicao}-{fim}"}
        )
        with urllib.request.urlopen(pedido, timeout=120) as resposta:
            dados = resposta.read()
        self.requisicoes += 1
        self.bytes_lidos += len(dados)
        self.posicao += len(dados)
        return dados


print(f"Abrindo {URL} ...")
inicio = time.perf_counter()
remoto = ArquivoRemoto(URL)
print(f"  tamanho: {remoto.tamanho / 1024**3:.1f} GB")

try:
    arquivo = zipfile.ZipFile(remoto)
except Exception as erro:                              # noqa: BLE001
    print(f"FALHA ao abrir como zip: {type(erro).__name__}: {erro}")
    raise SystemExit(1)

decorrido = time.perf_counter() - inicio
nomes = arquivo.namelist()
print(f"  diretorio central lido em {decorrido:.1f}s")
print(f"  requisicoes: {remoto.requisicoes}, baixados: {remoto.bytes_lidos / 1024**2:.1f} MB")
print(f"  entradas: {len(nomes)}")
print("\nPrimeiras 8 entradas:")
for nome in nomes[:8]:
    info = arquivo.getinfo(nome)
    print(f"   {nome}   ({info.file_size} bytes, compress_type={info.compress_type})")

print("\nAlgum nome parece 'bus_02020.png'?")
candidatos = [n for n in nomes[:200000] if "bus_" in n.lower()]
print(f"   entradas contendo 'bus_': {len(candidatos)}")
if candidatos:
    print(f"   exemplos: {candidatos[:3]}")

antes = remoto.bytes_lidos
alvo = nomes[0]
dados = arquivo.read(alvo)
print(f"\nLeitura de uma entrada ({alvo}): {len(dados)} bytes")
print(f"  custo em rede: {(remoto.bytes_lidos - antes) / 1024:.1f} KB")
