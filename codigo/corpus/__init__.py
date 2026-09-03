"""Montagem e particionamento do corpus.

`manifesto.py` e o codigo que decide QUAL imagem vai para qual split -- e,
portanto, a espinha do registro de reproducao. Cada linha do manifesto grava
caminho absoluto, rotulo, gerador, fonte, split e dimensoes.

Nota: ate 03/09/2026 este pacote se chamava `codigo/data/` e, por causa disso,
era engolido pela regra `data/` do .gitignore -- 590 linhas de codigo fora do
versionamento sem que nada avisasse. O nome atual o traz para dentro do git.

As perturbacoes do protocolo de robustez ficam em
`codigo/preprocessamento/perturbacoes.py`, junto do restante do que se aplica
a imagem antes das tecnicas.
"""
from .manifesto import (
    ManifestEntry, as_arrays, assign_splits, format_report, index_corpus,
    ood_split, read_manifest, verify_integrity, write_manifest,
)

__all__ = [
    "ManifestEntry", "as_arrays", "assign_splits", "format_report", "index_corpus",
    "ood_split", "read_manifest", "verify_integrity", "write_manifest",
]
