import numpy as np
import xarray as xr

from probench.scenarios.scenario_base import PerfScenario

class PR7736Scenario(PerfScenario):
    """
    PR #7736 — Avoid reindexing when join is exact and copy is False

    Parameters:
    - ny: Number of years
    - no: Number of objects to align
    """

    def setup(self):
        params = self.params
        n_years = int(params["ny"])
        n_objects = int(params["no"])

        time_coords = xr.cftime_range(
            start="0001-01-01",
            periods=n_years * 365,  # Daily data for n_years
            freq="D",
            calendar="noleap"
        )

        nx, ny_spatial = 50, 50

        arrays = []
        for i in range(n_objects):
            da = xr.DataArray(
                self.rng.standard_normal((len(time_coords), nx, ny_spatial)),
                dims=["time", "x", "y"],
                coords={"time": time_coords},
            )
            arrays.append(da)

        self.args = {"arrays": arrays}

    def run(self):
        return xr.align(*self.args["arrays"], join= "exact", copy= False)
