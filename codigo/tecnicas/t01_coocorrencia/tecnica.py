"""T01 - CNN sobre matrizes de coocorrencia RGB (Nataraj et al., 2019).

Para cada canal de cor (R, G e B) calcula-se uma matriz de coocorrencia sobre
as intensidades de pixels adjacentes, com deslocamento unitario, gerando uma
matriz 256 x 256 por canal. As tres matrizes sao empilhadas em um tensor
3 x 256 x 256, que preserva correlacoes de segunda ordem entre intensidades
vizinhas e constitui a entrada da rede.

A rede reproduz a arquitetura original: seis camadas convolucionais com 32, 32,
64, 64, 128 e 128 filtros, alternando nucleos 3x3 e 5x5, ativacao ReLU e tres
camadas de max pooling intercaladas, seguidas de duas camadas totalmente
conectadas de 256 unidades e de uma camada de saida com funcao sigmoide.
A rede e treinada do zero, sem pesos pre-treinados.
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from ...configuracao import T01, T01Config
from ...sementes import seed_worker
from ..base import BaseTechnique, TechniqueError


# ---------------------------------------------------------------------------
# Extracao de caracteristicas
# ---------------------------------------------------------------------------


def cooccurrence_matrix(channel: np.ndarray, offset: tuple[int, int], levels: int = 256) -> np.ndarray:
    """Matriz de coocorrencia de um canal para um deslocamento (dy, dx).

    O par (i, j) acumula a contagem de ocorrencias em que um pixel de
    intensidade ``i`` tem, na posicao deslocada, um pixel de intensidade ``j``.
    A matriz e normalizada para somar 1, tornando-a invariante a resolucao.
    """
    if channel.ndim != 2:
        raise ValueError(f"esperado canal 2-D, obtido shape {channel.shape}")

    dy, dx = offset
    height, width = channel.shape
    if abs(dy) >= height or abs(dx) >= width:
        raise ValueError("deslocamento maior que as dimensoes do canal")

    # Recorta as duas vistas deslocadas sem copiar os dados.
    y0_a, y1_a = max(0, -dy), height - max(0, dy)
    x0_a, x1_a = max(0, -dx), width - max(0, dx)
    reference = channel[y0_a:y1_a, x0_a:x1_a]
    shifted = channel[y0_a + dy:y1_a + dy, x0_a + dx:x1_a + dx]

    flat_index = reference.astype(np.int64).ravel() * levels + shifted.astype(np.int64).ravel()
    counts = np.bincount(flat_index, minlength=levels * levels)
    matrix = counts.reshape(levels, levels).astype(np.float32)

    total = matrix.sum()
    if total > 0:
        matrix /= total
    return matrix


def rgb_cooccurrence_tensor(
    image: Image.Image,
    offset: tuple[int, int] = T01.offset,
    levels: int = T01.levels,
) -> np.ndarray:
    """Empilha as matrizes de coocorrencia dos canais R, G e B.

    Retorna um tensor de forma (3, levels, levels), na ordem de canais RGB.
    """
    array = np.asarray(image.convert("RGB"), dtype=np.uint8)
    planes = [cooccurrence_matrix(array[:, :, c], offset, levels) for c in range(3)]
    return np.stack(planes, axis=0)


class CooccurrenceDataset(Dataset):
    """Calcula a coocorrencia sob demanda, evitando materializar o corpus.

    Cada tensor 3 x 256 x 256 em float32 ocupa 768 KB; manter as 360.000
    imagens em memoria seria inviavel no hardware descrito na Secao 3.2.
    """

    def __init__(
        self,
        paths: Sequence[Path],
        labels: np.ndarray | None = None,
        config: T01Config = T01,
        scale: float = 1e4,
    ) -> None:
        self.paths = list(paths)
        self.labels = None if labels is None else np.asarray(labels, dtype=np.float32)
        self.config = config
        # As matrizes normalizadas concentram-se em valores muito pequenos
        # (~1/65536). O reescalonamento evita gradientes proximos de zero nas
        # primeiras camadas sem alterar a informacao relativa.
        self.scale = scale

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int):
        with Image.open(self.paths[index]) as image:
            tensor = rgb_cooccurrence_tensor(image, self.config.offset, self.config.levels)
        features = torch.from_numpy(tensor * self.scale)
        if self.labels is None:
            return features
        return features, torch.tensor(self.labels[index])


# ---------------------------------------------------------------------------
# Arquitetura
# ---------------------------------------------------------------------------


class CooccurrenceCNN(nn.Module):
    """Rede convolucional de Nataraj et al. (2019)."""

    def __init__(self, in_channels: int = 3) -> None:
        super().__init__()
        # Filtros 32, 32, 64, 64, 128, 128 alternando nucleos 3x3 e 5x5,
        # com tres camadas de max pooling intercaladas.
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=5, padding=2), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),                                             # 256 -> 128
            nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=5, padding=2), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),                                             # 128 -> 64
            nn.Conv2d(64, 128, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=5, padding=2), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),                                             # 64 -> 32
        )
        # Reduz o mapa 128 x 32 x 32 a um vetor de 128 valores, mantendo as
        # duas camadas densas de 256 unidades do trabalho original.
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 256), nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, 256), nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Retorna o logit; a sigmoide e aplicada na inferencia."""
        return self.classifier(self.pool(self.features(x))).squeeze(1)


# ---------------------------------------------------------------------------
# Tecnica
# ---------------------------------------------------------------------------


class T01Cooccurrence(BaseTechnique):
    technique_id = "T01"
    trainable = True

    def __init__(self, device: str | None = None, config: T01Config = T01, seed: int = 42) -> None:
        resolved = device or ("cuda" if torch.cuda.is_available() else "cpu")
        super().__init__(device=resolved)
        self.config = config
        self.seed = seed
        self.model = CooccurrenceCNN().to(self.device)
        self.history: list[dict[str, float]] = []

    # -- caracteristicas ------------------------------------------------

    def extract_features(self, paths: Sequence[Path]) -> np.ndarray:
        """Tensores de coocorrencia empilhados, shape (n, 3, 256, 256)."""
        out = []
        for path in paths:
            with Image.open(path) as image:
                out.append(rgb_cooccurrence_tensor(image, self.config.offset, self.config.levels))
        return np.stack(out, axis=0)

    # -- treinamento ----------------------------------------------------

    def fit(
        self,
        paths: Sequence[Path],
        y: np.ndarray,
        val_paths: Sequence[Path] | None = None,
        val_y: np.ndarray | None = None,
        num_workers: int = 4,
        class_weight: float | None = None,
    ) -> "T01Cooccurrence":
        """Treina a rede do zero com parada antecipada pela AUC de validacao.

        A parada antecipada e ativada apos ``early_stopping_patience`` epocas
        consecutivas sem melhora da AUC no conjunto de validacao, com
        restauracao dos pesos da epoca de melhor desempenho.
        """
        from sklearn.metrics import roc_auc_score

        torch.manual_seed(self.seed)
        y = np.asarray(y, dtype=np.float32)

        train_loader = DataLoader(
            CooccurrenceDataset(paths, y, self.config),
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=num_workers,
            worker_init_fn=seed_worker,
            generator=torch.Generator().manual_seed(self.seed),
            pin_memory=self.device.startswith("cuda"),
            drop_last=False,
        )

        val_loader = None
        if val_paths is not None and val_y is not None:
            val_loader = DataLoader(
                CooccurrenceDataset(val_paths, np.asarray(val_y, dtype=np.float32), self.config),
                batch_size=self.config.batch_size,
                shuffle=False,
                num_workers=num_workers,
                pin_memory=self.device.startswith("cuda"),
            )

        # pos_weight compensa desbalanceamento residual entre geradores (R07).
        pos_weight = None if class_weight is None else torch.tensor([class_weight], device=self.device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config.learning_rate)

        best_auc = -np.inf
        best_state = copy.deepcopy(self.model.state_dict())
        epochs_without_improvement = 0
        self.history = []

        for epoch in range(self.config.max_epochs):
            self.model.train()
            running_loss, n_seen = 0.0, 0
            for features, targets in train_loader:
                features = features.to(self.device, non_blocking=True)
                targets = targets.to(self.device, non_blocking=True)

                optimizer.zero_grad(set_to_none=True)
                loss = criterion(self.model(features), targets)
                loss.backward()
                optimizer.step()

                running_loss += loss.item() * features.size(0)
                n_seen += features.size(0)

            epoch_record = {"epoch": epoch, "train_loss": running_loss / max(n_seen, 1)}

            if val_loader is None:
                self.history.append(epoch_record)
                continue

            val_probs = self._infer_loader(val_loader, has_labels=True)
            val_auc = float(roc_auc_score(np.asarray(val_y), val_probs))
            epoch_record["val_auc"] = val_auc
            self.history.append(epoch_record)

            if val_auc > best_auc:
                best_auc = val_auc
                best_state = copy.deepcopy(self.model.state_dict())
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
                if epochs_without_improvement >= self.config.early_stopping_patience:
                    break

        # Restaura os pesos da epoca de melhor AUC de validacao.
        self.model.load_state_dict(best_state)
        self._fitted = True
        return self

    # -- inferencia -----------------------------------------------------

    @torch.no_grad()
    def _infer_loader(self, loader: DataLoader, has_labels: bool) -> np.ndarray:
        self.model.eval()
        chunks = []
        for batch in loader:
            features = batch[0] if has_labels else batch
            features = features.to(self.device, non_blocking=True)
            chunks.append(torch.sigmoid(self.model(features)).float().cpu().numpy())
        if not chunks:
            return np.empty(0, dtype=np.float64)
        return np.concatenate(chunks).astype(np.float64)

    def predict_proba(self, paths: Sequence[Path], num_workers: int = 2) -> np.ndarray:
        self._check_fitted()
        loader = DataLoader(
            CooccurrenceDataset(paths, None, self.config),
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=self.device.startswith("cuda"),
        )
        return self._infer_loader(loader, has_labels=False)

    # -- persistencia ---------------------------------------------------

    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {"state_dict": self.model.state_dict(), "seed": self.seed, "history": self.history},
            path,
        )

    def load(self, path: Path) -> "T01Cooccurrence":
        path = Path(path)
        if not path.exists():
            raise TechniqueError(f"T01: checkpoint nao encontrado em {path}")
        payload = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(payload["state_dict"])
        self.model.to(self.device)
        self.history = payload.get("history", [])
        self._fitted = True
        return self
