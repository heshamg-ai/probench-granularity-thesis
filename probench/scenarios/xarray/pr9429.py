import cftime
import xarray as xr

from probench.scenarios.scenario_base import PerfScenario

class PR9429Scenario(PerfScenario):
    """
    PR #9429 — Benchmark groupby setup with cftime auxiliary coordinates.

    Parameters:
    - ny: Number of y points 
    - nx: Number of x points 
    - cf: Use cftime
    """

    def setup(self):
        params = self.params
        n_y = int(params["ny"])
        n_x = int(params["nx"])
        n_time = 365 * 30

        arr = self.rng.standard_normal((n_y, n_x, n_time))
        time = xr.date_range("2000", periods=n_time, use_cftime="1")

        as_da = xr.DataArray(time)
        labeled_time = []
        for year, month in zip(as_da.dt.year.values, as_da.dt.month.values):
            labeled_time.append(cftime.datetime(int(year), int(month), 1))

        da = xr.DataArray(
            arr,
            dims=("y", "x", "time"),
            coords={"time": time, "time2": ("time", labeled_time)},
        )

        self.args = {"da": da}

    def run(self):
        return self.args["da"].groupby("time.month")
