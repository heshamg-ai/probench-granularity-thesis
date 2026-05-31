import numpy as np
import pandas as pd
from probench.scenarios.scenario_base import PerfScenario


class PR54835Scenario(PerfScenario):
    """
    PR #54835_1 — DataFrame.sort_values with multiple keys.

    Parameters:
    - N: Number of rows
    - LV: Number of MultiIndex levels
    """

    def setup(self):
        params = self.params
        N = int(params["N"])
        LV = int(params["LV"])

        arrays = [self.rng.integers(0, 10_000, size=N) for _ in range(LV)]
        names = [f"level_{i}" for i in range(LV)]
        mi = pd.MultiIndex.from_arrays(arrays, names=names)

        df = pd.DataFrame({"value": self.rng.standard_normal(N)}, index=mi)
        self.args = {"df": df}

    def run(self):
        return self.args["df"].sort_index()
