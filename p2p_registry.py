"""P2P accessibility registry for multi-GPU DLPack operations.

Caches CUDA or HIP peer-access queries per GPU pair to avoid repeated runtime
API calls.
"""

import ctypes
import logging
import torch

logger = logging.getLogger("MultiGPU")

_runtime_library = None


def _get_runtime_peer_query():
    """Load the active CUDA or HIP peer-access API once."""
    global _runtime_library
    if _runtime_library is not None:
        return _runtime_library

    if getattr(torch.version, "hip", None):
        library_names = ("libamdhip64.so", "libamdhip64.so.6")
        function_name = "hipDeviceCanAccessPeer"
    else:
        library_names = ("libcudart.so",)
        function_name = "cudaDeviceCanAccessPeer"

    for library_name in library_names:
        try:
            library = ctypes.CDLL(library_name)
            _runtime_library = (library, getattr(library, function_name), function_name)
            return _runtime_library
        except (AttributeError, OSError):
            continue

    raise OSError(f"Unable to load {function_name} from {library_names}")


class MultiGPUP2PRegistry:
    """Cached registry for CUDA/HIP peer-to-peer accessibility between GPUs."""

    def __init__(self):
        self._cache = {}

    @staticmethod
    def _raw_can_access_peer(device_a: int, device_b: int) -> bool:
        """Call the active CUDA/HIP peer-access API via ctypes."""
        _, peer_query, function_name = _get_runtime_peer_query()
        can_access = ctypes.c_int(0)
        result = peer_query(ctypes.byref(can_access), device_a, device_b)
        if result != 0:
            logger.warning(
                f"[MultiGPU P2P] {function_name}({device_a}, {device_b}) "
                f"returned error code {result}, assuming no P2P"
            )
            return False
        return bool(can_access.value)

    def can_access_peer(self, src_device: int, dst_device: int) -> bool:
        """Check if src_device can access dst_device memory via P2P.

        Results are cached per (src, dst) pair.
        """
        if src_device == dst_device:
            return True

        key = (src_device, dst_device)
        if key not in self._cache:
            if not torch.cuda.is_available():
                self._cache[key] = False
            else:
                try:
                    result = self._raw_can_access_peer(src_device, dst_device)
                except OSError as exc:
                    logger.warning(
                        f"[MultiGPU P2P] peer-access API unavailable: {exc}; "
                        "using CPU staging"
                    )
                    result = False
                self._cache[key] = result
                logger.info(
                    f"[MultiGPU P2P] can_access_peer({src_device}, {dst_device}) = {result}"
                )
        return self._cache[key]

    def clear_cache(self):
        """Clear the P2P cache (useful for testing)."""
        self._cache.clear()


# Module-level singleton
p2p_registry = MultiGPUP2PRegistry()
