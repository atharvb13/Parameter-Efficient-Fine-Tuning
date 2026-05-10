"""
ASDL uses torch.cuda.nvtx even on CPU-only builds where range_push fails.
Patch push/pop to no-ops when CUDA NVTX is unavailable (e.g. macOS / CPU).
"""


def patch_nvtx_if_needed():
    import torch.cuda.nvtx as nvtx

    try:
        nvtx.range_push("__asdl_probe__")
        nvtx.range_pop()
    except RuntimeError:
        nvtx.range_push = lambda msg: 0  # noqa: ARG005
        nvtx.range_pop = lambda: None
