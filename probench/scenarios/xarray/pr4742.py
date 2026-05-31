import numpy as np
import xarray as xr

from probench.scenarios.scenario_base import PerfScenario

class PR4742Scenario(PerfScenario):
    """
    PR #4742 — Benchmark DataArray._ipython_key_completions_

    Parameters:
    - nd: Number of dimensions
    - nc: Number of coordinates
    """

    def setup(self):
        params = self.params
        n_dims = int(params["nd"])
        n_coords = int(params["nc"])
        n_variables = 20

        dim_sizes = {f"dim_{i}": 5 for i in range(n_dims)}
        dims = list(dim_sizes.keys())
        shape = tuple(dim_sizes.values())

        coords = {dim: np.arange(size) for dim, size in dim_sizes.items()}
        for c in range(n_coords - n_dims):
            coord_dim = dims[c % n_dims]
            coords[f"coord_{c}"] = (coord_dim, np.arange(dim_sizes[coord_dim]))

        data_vars = {
            f"var_{v}": (dims, self.rng.standard_normal(shape))
            for v in range(n_variables)
        }

        ds = xr.Dataset(data_vars=data_vars, coords=coords)

        self.args = {"ds": ds}

    def run(self):
        return self.args["ds"]._ipython_key_completions_()
