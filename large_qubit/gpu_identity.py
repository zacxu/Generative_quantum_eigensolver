"""Canonical NVIDIA GPU identity without importing CUDA or loading a model."""

import re
from uuid import UUID


_COMPLETE_UUID = re.compile(
    r'[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-'
    r'[0-9a-fA-F]{4}-[0-9a-fA-F]{12}'
)


def normalize_gpu_uuid(value):
    """Accept a complete PyTorch/NVIDIA UUID and return ``GPU-<lowercase UUID>``.

    PyTorch's native ``_CUuuid`` string contains the 128-bit label without the
    NVIDIA ``GPU-`` prefix. CUDA visibility masks and nvidia-smi use that prefix.
    Partial, missing, zero and MIG identities are deliberately not accepted.
    """
    text = str(value)
    label = text[4:] if text[:4].upper() == 'GPU-' else text
    if not _COMPLETE_UUID.fullmatch(label):
        raise ValueError(f'A complete nonzero GPU UUID is required; received {text!r}')
    parsed = UUID(label)
    if parsed.int == 0:
        raise ValueError(f'A complete nonzero GPU UUID is required; received {text!r}')
    return 'GPU-' + str(parsed)
