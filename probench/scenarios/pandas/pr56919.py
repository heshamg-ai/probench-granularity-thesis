import pandas as pd
from probench.scenarios.scenario_base import PerfScenario
import numpy as np

class PR56919Scenario(PerfScenario):
    """
    PR #56919 – Benchmark `DataFrame.join` with `how="left"` (or `"right"`) and `sort=True`.
    
    Parameters:
    - N :Index Length
    """

    def setup(self):
        params = self.params
        N = int(params["N"])

        data = [f"i-{i}" for i in range(N)]
        shift = max(1, int(N * 0.5 / 100)) # create an misalginment

        idx1 = pd.Index(data, dtype="string[pyarrow_numpy]")
        idx2 = pd.Index(data[shift:], dtype="string[pyarrow_numpy]")

        left = pd.DataFrame({"v1": np.ones(len(idx1), dtype=np.int64)}, index=idx1)
        right = pd.DataFrame({"v2": np.ones(len(idx2), dtype=np.int64)}, index=idx2)

        self.args = {"left": left, "right": right}

    def run(self):
        return self.args["left"].join(self.args["right"], how="left", sort=True)
