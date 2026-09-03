"""Modulos de deteccao T01-T05 com interface unificada.

Cada tecnica vive numa pasta propria, cujo __init__.py traz a ficha dela:
o que mede, em que resolucao foi medida, o resultado e a fragilidade
conhecida. A implementacao fica em `tecnica.py`.

    t01_coocorrencia/   coocorrencia RGB + CNN      AUC 0,9962
    t02_spai/           aprendizado espectral       AUC 0,9976
    t03_benford/        Benford sobre DCT           AUC 0,8064
    t04_geometria/      geometria projetiva         AUC 0,5333 (0,6225 no benchmark)
    t05_fusao/          fusao tardia (contribuicao) AUC 0,9996

As importacoes de T01 sao adiadas porque exigem PyTorch, que pode nao estar
instalado em ambientes voltados apenas a T03/T05.
"""
from .base import BaseTechnique, TechniqueError, TechniqueResult
from .t02_spai import T02SPAI
from .t03_benford import T03Benford
from .t04_geometria import T04ProjectiveGeometry
from .t05_fusao import T05Fusion, build_score_matrix

__all__ = [
    "BaseTechnique", "TechniqueError", "TechniqueResult",
    "T01Cooccurrence", "T02SPAI", "T03Benford", "T04ProjectiveGeometry",
    "T05Fusion", "build_score_matrix",
]


def __getattr__(name):
    if name == "T01Cooccurrence":
        from .t01_coocorrencia import T01Cooccurrence
        return T01Cooccurrence
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
