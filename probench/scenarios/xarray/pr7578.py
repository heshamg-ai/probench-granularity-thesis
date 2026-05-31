import numpy as np
import xarray as xr

from probench.scenarios.scenario_base import PerfScenario

class PR7578Scenario(PerfScenario):
    """
    PR #7578 — Benchmark DatasetRolling.construct with stride on Datasets without indexes
    
    Parameters:
    - nt: Size of rolling dimension
    - str: Stride for construct
    """

    def setup(self):
        params = self.params

        n_time = int(params["nt"])
        stride = int(params["str"])
        n_spatial = 100
        has_non_dim_coord = True  

        data_vars = {
            "temperature": (["time", "space"], self.rng.standard_normal((n_time, n_spatial))),
            "pressure": (["time", "space"], self.rng.standard_normal((n_time, n_spatial))),
        }

        coords = {
            "time": np.arange(n_time),
            "space": np.arange(n_spatial),
        }

        if has_non_dim_coord:
            coords["time_label"] = ("time", [f"t_{i}" for i in range(n_time)])
            coords["space_weight"] = ("space", self.rng.random(n_spatial))

        ds = xr.Dataset(data_vars=data_vars, coords=coords)

        self.args = {
            "func": ds.rolling(time=10),
            "stride": stride,
        }

    def run(self):
        return self.args["func"].construct("window_dim", stride=self.args["stride"])
