import numpy as np
import pandas as pd
import xarray as xr

from probench.scenarios.scenario_base import PerfScenario

class PR7382Scenario(PerfScenario):
    """
    PR #7382 — Benchmark for Dataset.assign in alignment between indexed and non-indexed objects of the same shape

    Parameters:
    - n: Size of coordinate dimension
    """

    def setup(self):
        params = self.params
        n = int(params["n"])

        mi = pd.MultiIndex.from_arrays([
            np.repeat(np.arange(n // 10), 10),
            np.tile(np.arange(10), n // 10),
        ], names=["level1", "level2"])

        ds = xr.Dataset(
            {
                "d1": ("x", self.rng.standard_normal(n)),
                "d2": ("x", self.rng.standard_normal(n)),
                "d3": ("x", self.rng.random(n) > 0.5),
            },
            coords={"x": mi}
        )
       
        self.args = {"ds": ds}
       

    def run(self):
        return self.args["ds"].assign(foo=~self.args["ds"]["d3"])
