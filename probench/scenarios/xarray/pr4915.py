import numpy as np
import xarray as xr

from probench.scenarios.scenario_base import PerfScenario

class PR4915Scenario(PerfScenario):
    """
    PR #4915 — Benchmark Rolling.mean 

    Parameters:
    - nt: Size of rolling dimension
    - win: Rolling window size
    """

    def setup(self):
        params = self.params

        n_time = int(params["NT"])
        window = int(params["W"])
        n_spatial = 500
        data = self.rng.standard_normal((n_time, n_spatial))

        data_array = xr.DataArray(
            data,
            dims=["time", "space"],
            coords={"time": np.arange(n_time), "space": np.arange(n_spatial)},
            attrs={"units": "test", "description": "benchmark data"},
        )

        func = data_array.rolling(time=window)

        self.args = {
            "func": func
        }

    def run(self):
        return self.args["func"].mean()
