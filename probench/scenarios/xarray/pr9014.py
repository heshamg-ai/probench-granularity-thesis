import os
import tempfile

import numpy as np
import xarray as xr
from xarray.core.datatree import DataTree
from xarray.backends.netCDF4_ import NetCDF4BackendEntrypoint

from probench.scenarios.scenario_base import PerfScenario

class PR9014Scenario(PerfScenario):
    """
    PR #9014 — Benchmark open_datatree on NetCDF files.

    Parameters:
    - ng: Number of groups in hierarchical structure
    - nv: Number of variables per group
    - sz: Size of data arrays
    """

    def setup(self):
        params = self.params
        n_groups = int(params["ng"])
        n_vars_per_group = int(params["nv"])
        array_size = int(params["sz"])

        tree_dict = {"/": xr.Dataset()}

        for g in range(n_groups):
            group_name = f"/group_{g}"
            data_vars = {}

            for v in range(n_vars_per_group):
                data_vars[f"var_{v}"] = (["x"], self.rng.standard_normal(array_size))

            tree_dict[group_name] = xr.Dataset(
                data_vars=data_vars,
                coords={"x": np.arange(array_size)},
            )

        dt = DataTree.from_dict(tree_dict)

        tmpdir = tempfile.mkdtemp()
        filepath = os.path.join(tmpdir, "test.nc")
        dt.to_netcdf(filepath, engine="netcdf4")

        self.args = {"filepath": filepath}

    def run(self):
        return NetCDF4BackendEntrypoint().open_datatree(self.args["filepath"])
