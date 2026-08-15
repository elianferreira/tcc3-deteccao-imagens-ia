"""Modulos de deteccao T01-T05 com interface unificada.

As importacoes de T01 sao adiadas porque exigem PyTorch, que pode nao estar
instalado em ambientes voltados apenas a T03/T05.
"""
from .base import BaseTechnique, TechniqueError, TechniqueResult
from .t02_spai import T02SPAI
from .t03_benford import T03Benford
from .t04_geometry import T04ProjectiveGeometry
from .t05_fusion import T05Fusion, build_score_matrix

__all__ = [
    "BaseTechnique", "TechniqueError", "TechniqueResult",
    "T01Cooccurrence", "T02SPAI", "T03Benford", "T04ProjectiveGeometry",
    "T05Fusion", "build_score_matrix",
]


def __getattr__(name):
    if name == "T01Cooccurrence":
        from .t01_cooccurrence import T01Cooccurrence
        return T01Cooccurrence
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
