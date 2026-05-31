import numpy as np
import pandas as pd
import xarray as xr

from probench.scenarios.scenario_base import PerfScenario

class PR10316Scenario(PerfScenario):
    """
    PR #10316 — Benchmark VectorizedIndexer

    Parameters:
    - ni: Number of indices in the vectorized indexer (key parameter - larger = more savings)
    - nd: Number of dimensions in the indexer pattern (1=1d, 2=2d)
    """

    def setup(self):
        params = self.params
        n_index = int(params["ni"])  # Number of indices to select
        n_dims = int(params["nd"])   # 1 for 1D indexer, 2 for 2D indexer

        nx, ny, nt = 1000, 500, 200

        ds = xr.Dataset(
            {
                "var1": (("x", "y"), self.rng.standard_normal((nx, ny))),
                "var2": (("x", "t"), self.rng.standard_normal((nx, nt))),
                "var3": (("y", "t"), self.rng.standard_normal((ny, nt))),
                "var4": (("x", "y"), self.rng.standard_normal((nx, ny))),
                "var5": (("x", "t"), self.rng.standard_normal((nx, nt))),
                "var6": (("y", "t"), self.rng.standard_normal((ny, nt))),
            },
            coords={
                "x": np.arange(nx),
                "y": np.linspace(0, 1, ny),
                "t": pd.date_range("1970-01-01", periods=nt, freq="D"),
            },
        )

        # Create vectorized indexers — all 3 dims indexed to maximize avoided copies
        if n_dims == 1:
            # 1D vectorized indexer
            x_idx = xr.DataArray(
                self.rng.integers(0, nx, size=n_index, dtype=np.int64), dims=["a"]
            )
            y_idx = xr.DataArray(
                self.rng.integers(0, ny, size=n_index, dtype=np.int64), dims=["a"]
            )
            t_idx = xr.DataArray(
                self.rng.integers(0, nt, size=n_index, dtype=np.int64), dims=["a"]
            )
        else:
            # 2D vectorized indexer
            grid_size = int(np.sqrt(n_index))
            x_idx = xr.DataArray(
                self.rng.integers(0, nx, size=(grid_size, grid_size), dtype=np.int64), dims=["a", "b"]
            )
            y_idx = xr.DataArray(
                self.rng.integers(0, ny, size=(grid_size, grid_size), dtype=np.int64), dims=["a", "b"]
            )
            t_idx = xr.DataArray(
                self.rng.integers(0, nt, size=(grid_size, grid_size), dtype=np.int64), dims=["a", "b"]
            )

        self.args = {
            "ds": ds,
            "x_idx": x_idx,
            "y_idx": y_idx,
            "t_idx": t_idx,
        }

    def run(self):
        # The optimization happens inside isel — index all 3 dims to maximize benefit
        return self.args["ds"].isel(x=self.args["x_idx"], y=self.args["y_idx"], t=self.args["t_idx"])
