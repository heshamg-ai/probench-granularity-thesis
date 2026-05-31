import numpy as np
import pandas as pd
from probench.scenarios.scenario_base import PerfScenario


class PR52430Scenario(PerfScenario):
    """
    PR #52430 – Benchmark Series.to_numpy with nullable float64 dtype.

    Parameters:
    - N: Series length
    - NAP: NA percentage
    """

    def setup(self):
        params = self.params
        N = int(params["N"])
        NAP = float(params["NAP"])
        
        series = pd.Series(self.rng.random(N), dtype="float64")

        # Add NA values if requested
        if NAP > 0:
            n_na = int(N * NAP)
            na_indices = self.rng.choice(N, size=n_na, replace=False)
            series.iloc[na_indices] = pd.NA

        self.args = {"series": series}

    def run(self):
        return self.args["series"].to_numpy(dtype="float64", na_value=np.nan)
