import numpy as np
import xarray as xr

from probench.scenarios.scenario_base import PerfScenario

class PR3533Scenario(PerfScenario):
    """
    PR #3533 — Benchmark DataArray.isel when indexed by int on small arrays

    Parameters:
    - nd: Number of dimensions
    - sz: Size of each dimension
    """

    def setup(self):
        params = self.params
        n_dims = int(params["nd"])
        dim_size = int(params["sz"])

        dims = [f"dim_{i}" for i in range(n_dims)]
        shape = tuple([dim_size] * n_dims)
        coords = {dim: np.arange(dim_size) for dim in dims}

        da = xr.DataArray(
            self.rng.standard_normal(shape),
            dims=dims,
            coords=coords,
        )

        self.args = {"da": da}

    def run(self):
        return self.args["da"].isel(dim_0=0)
