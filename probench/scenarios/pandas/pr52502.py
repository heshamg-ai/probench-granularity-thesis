import numpy as np
import pandas as pd
from probench.scenarios.scenario_base import PerfScenario


class PR52502Scenario(PerfScenario):
    """
    PR #52502 – Benchmark Series.corr with Float64 dtype.

    Parameters:
    - N: Series length
    """

    def setup(self):
        params = self.params

        N = int(params["N"])

        s1 = pd.Series(self.rng.random(N), dtype="Float64")
        s2 = pd.Series(self.rng.random(N), dtype="Float64")

        self.args = {"s1": s1, "s2": s2}

    def run(self):
        return self.args["s1"].corr(self.args["s2"])
