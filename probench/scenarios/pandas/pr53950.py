import pandas as pd
import pyarrow as pa
from probench.scenarios.scenario_base import PerfScenario


class PR53950Scenario(PerfScenario):
    """
    PR #53950 – Benchmark `Series.ffill` with PyArrow-backed extension dtypes.

    Parameters:
    - N: Size
    """

    def setup(self):
        params = self.params
        N = int(params["N"])
        base_data = self.rng.integers(0, 1000, size=N)
        dtype = pd.ArrowDtype(pa.int64())

        values = [pd.NA] * N
        valid_indices = self.rng.choice(N, size=N // 2, replace=False)
        for idx in valid_indices:
            values[idx] = base_data[idx]

        self.args = {
            "ser": pd.Series(values, dtype=dtype)
        }

    def run(self):
        return self.args["ser"].ffill()
