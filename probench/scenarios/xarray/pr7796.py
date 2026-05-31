import numpy as np
import xarray as xr

from probench.scenarios.scenario_base import PerfScenario

class PR7796Scenario(PerfScenario):
    """
    PR #7796 — Benchmark .dt accessor with CFTimeIndex

    Parameters:
    - nt: Number of time points
    """

    def setup(self):
        params = self.params
        n_time = int(params["nt"])

        time = xr.date_range("2000", periods=n_time, calendar="standard")
        data = np.ones((n_time,))

        data_array = xr.DataArray(
            data,
            dims="time",
            coords={"time": time})

        self.args = {
            "data_array": data_array,
            "dt_property": params["prop"]
        }

    def run(self):
        return getattr(self.args["data_array"].time.dt, self.args["dt_property"])
