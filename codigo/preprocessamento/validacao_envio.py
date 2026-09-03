"""Validacao do arquivo enviado, antes de qualquer processamento (RF06).

RN01  formatos aceitos: PNG, JPEG, WebP
RN02  tamanho maximo de 10 MB

O formato e confirmado pelo CONTEUDO do arquivo, nao pela extensao. Isso
nao e zelo teorico: o corpus tem um caso real -- as imagens de
`fake/adobe_firefly` tem extensao .png e sao JPEG por dentro. Conferir so
a extensao aceitaria arquivos renomeados e recusaria arquivos validos.
"""

from __future__ import annotations

from pathlib import Path

from PIL import Image

# RN01 e RN02
ALLOWED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp"}
ALLOWED_FORMATS = {"PNG", "JPEG", "WEBP"}
MAX_FILE_BYTES = 10 * 1024 * 1024




def validate_upload(path: str | None) -> tuple[bool, str]:
    """Valida formato e tamanho antes do envio ao pipeline.

    O formato e confirmado pelo conteudo do arquivo, e nao apenas pela
    extensao, de modo a rejeitar arquivos renomeados.
    """
    if not path:
        return False, "Nenhuma imagem foi enviada."

    file_path = Path(path)
    if not file_path.exists():
        return False, "Arquivo nao encontrado."

    size = file_path.stat().st_size
    if size > MAX_FILE_BYTES:
        return False, (
            f"Arquivo de {size / 1024 / 1024:.1f} MB excede o limite de 10 MB (RN02)."
        )

    if file_path.suffix.lower() not in ALLOWED_EXTENSIONS:
        return False, (
            f"Extensao '{file_path.suffix}' nao suportada. "
            f"Formatos aceitos: PNG, JPEG e WebP (RN01)."
        )

    try:
        with Image.open(file_path) as image:
            detected = (image.format or "").upper()
    except Exception:                                   # noqa: BLE001
        return False, "Arquivo corrompido ou nao reconhecido como imagem."

    if detected not in ALLOWED_FORMATS:
        return False, (
            f"O conteudo do arquivo e do tipo {detected}, nao suportado. "
            f"Formatos aceitos: PNG, JPEG e WebP (RN01)."
        )
    return True, "ok"
