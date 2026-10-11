"""PRIVATE calibration artifact: the strong-library frontier ModelNew.

This wraps the official library baseline() (the honest performance frontier `--headroom` was
calibrated against). It is kept here for RECALIBRATION only — it is NOT the delivery positive
control (that is the certified Triton witness in solve.py) and is never installed into /app,
because importing the strong library is exactly what the task forbids the agent from doing.
"""
import reference_model_full as _rm


class ModelNew:
    def __init__(self, *init_args):
        self._init_args = list(init_args)

    def to(self, device):
        self._device = device
        return self

    def __call__(self, *inputs):
        return _rm.baseline(*inputs, init_args=self._init_args)
