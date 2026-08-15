"""Fixacao de sementes e modo deterministico.

Atende ao requisito RNF02: para a mesma imagem de entrada, com versoes de
bibliotecas fixadas e sementes definidas, o sistema deve produzir escores de
deteccao identicos em execucoes distintas.
"""

from __future__ import annotations

import os
import random

import numpy as np

from .config import SEED


def set_deterministic(seed: int = SEED) -> None:
    """Fixa as sementes de random, NumPy, PyTorch e CUDA e ativa o modo
    deterministico dos frameworks utilizados."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    # Exigido pelo cuBLAS para tornar deterministicas as operacoes de GEMM.
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

    random.seed(seed)
    np.random.seed(seed)

    try:
        import torch
    except ImportError:  # permite usar T03/T05 sem PyTorch instalado
        return

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except (AttributeError, RuntimeError):
        # warn_only nao existe em versoes antigas; a falha nao deve
        # interromper a execucao.
        pass


def seed_worker(worker_id: int) -> None:
    """Semeia cada worker do DataLoader de forma reproduzivel."""
    import torch

    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)
