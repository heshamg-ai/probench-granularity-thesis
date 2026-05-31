import numpy as np
import dask.array as da
import xarray as xr

from probench.scenarios.scenario_base import PerfScenario



class PR10554Scenario(PerfScenario):
    """
    PR #10554 — Benchmark Dataset.interp() when dataset contains non-numeric scalars

    Parameters:
    - ni: Number of points to interpolate
    """

    def setup(self):
        params = self.params
        n_interp_points = int(params["ni"])
        
        ds = xr.Dataset(
            data_vars={
                "variable_name": (
                    "time",
                    da.from_array(np.array(["test"], dtype=str), chunks=(1,)),
                )
            },
            coords={"time": ("time", np.array([0]))},
        )

        self.ds = ds
        self.new_time = np.linspace(0, 10, n_interp_points)

        self.args = {"time": self.new_time}

    def run(self):
        return self.ds.interp(**self.args)
