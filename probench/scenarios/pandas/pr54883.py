import numpy as np
import pandas as pd
from probench.scenarios.scenario_base import PerfScenario


class PR54883Scenario(PerfScenario):
    """
    PR #54883 – Benchmark DataFrame.sort_index on an already-monotonic MultiIndex.

    Parameters:
    - N : Number of rows in the DataFrame
    - L : Number of MultiIndex levels
    """

    def setup(self):
        params = self.params
        N = int(params["N"])
        L = int(params["L"])

        arrays = [np.sort(self.rng.integers(0, 1000, size=N)) for _ in range(L)]
        names = [f"level_{i}" for i in range(L)]
        mi = pd.MultiIndex.from_arrays(arrays, names=names)
        df = pd.DataFrame({"value": self.rng.standard_normal(N)}, index=mi)
        df = df.sort_index()

        self.args = {"df": df}

    def run(self):
        return self.args["df"].sort_index()
