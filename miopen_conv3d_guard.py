import logging

import torch
import torch.nn.functional as F

logger = logging.getLogger("MultiGPU")

_PATCH_MARKER = "_mgpu_miopen_conv3d_guard"


def _is_miopen_failure(error):
    return "miopen" in str(error).lower()


def register_miopen_conv3d_guard():
    """Retry conv3d outside MIOpen when its hipBLASLt GEMM path fails on ROCm."""
    if not getattr(torch.version, "hip", None):
        return False

    original = F.conv3d
    if getattr(original, _PATCH_MARKER, False):
        return False

    warned = False

    def guarded_conv3d(*args, **kwargs):
        nonlocal warned
        try:
            return original(*args, **kwargs)
        except RuntimeError as error:
            if not _is_miopen_failure(error):
                raise
            if not warned:
                warned = True
                logger.warning(
                    "[MultiGPU] MIOpen conv3d failed (%s); retrying without MIOpen",
                    error,
                )
            with torch.backends.cudnn.flags(enabled=False):
                return original(*args, **kwargs)

    setattr(guarded_conv3d, _PATCH_MARKER, True)
    F.conv3d = guarded_conv3d
    return True
