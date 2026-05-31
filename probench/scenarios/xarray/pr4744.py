import numpy as np
import xarray as xr

from probench.scenarios.scenario_base import PerfScenario

class PR4744Scenario(PerfScenario):
    """
    PR #4744 — Benchmark Dataset.__getitem__ for datasets with large number of variables

    Parameters:
    - nv: Number of variables
    - sz: Size of each variable's data array
    """

    def setup(self):
        params = self.params
        n_variables = int(params["nv"])
        array_size = int(params["sz"])

        data_vars = {
            f"var_{v}": (["x", "y"], self.rng.standard_normal((array_size, array_size)))
            for v in range(n_variables)
        }

        coords = {
            "x": np.arange(array_size),
            "y": np.arange(array_size),
            "z": ("x", np.arange(array_size)),
        }

        ds = xr.Dataset(data_vars=data_vars, coords=coords)
        variable = f"var_{n_variables // 2}"


        self.args = {
            "ds": ds,
            "variable": variable
        }

    def run(self):
        return self.args["ds"][self.args["variable"]]
