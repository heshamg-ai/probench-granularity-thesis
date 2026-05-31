import warnings
import numpy as np
import xarray as xr

from probench.scenarios.scenario_base import PerfScenario

class PR7824Scenario(PerfScenario):
    """
    PR #7824 — Benchmark concat when processing datasets with many variables

    Parameters:
    - nds: Number of datasets to concatenate
    - nv: Number of variables per dataset
    """

    def setup(self):

        params = self.params
        n_datasets = int(params["nds"])
        n_variables = int(params["nv"])

        datasets = []
        for d in range(n_datasets):

            data_vars = {
                f"var_{v}": (["time", "x", "y"], self.rng.standard_normal((1, 100, 100)))
                for v in range(n_variables)}

            ds = xr.Dataset(
                data_vars=data_vars,
                coords={
                    "time": [d],  
                    "x": np.arange(100),
                    "y": np.arange(100)})

            datasets.append(ds)

        self.args = {"datasets": datasets}

    def run(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FutureWarning)
            return xr.concat(self.args["datasets"], dim="time")
