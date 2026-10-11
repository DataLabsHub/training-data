"""PRIVATE calibration artifact: GB200 torch.compile frontier adapter.

This adapter is for recalibration/debug only. It intentionally wraps the naive
`reference_model_full.Model` with `torch.compile` so the adapter matches the
`compile_headroom`/`frontier_source = "compile"` calibration metadata in
task.toml. It is not the delivered positive control; `solve_witness.py` is.
"""
import torch
import reference_model_full as _rm


class ModelNew:
    def __init__(self, *init_args):
        self._model = _rm.Model(*init_args)
        self._compiled = None

    def to(self, device):
        if hasattr(self._model, "to"):
            self._model = self._model.to(device)
        self._device = device
        return self

    def __call__(self, *inputs):
        if self._compiled is None:
            def _run(*args):
                return self._model(*args)
            try:
                self._compiled = torch.compile(_run, dynamic=False)
            except Exception:
                self._compiled = _run
        return self._compiled(*inputs)
