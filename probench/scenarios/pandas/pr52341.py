import numpy as np
import pandas as pd
from probench.scenarios.scenario_base import PerfScenario


class PR52341Scenario(PerfScenario):
    """
    PR #52341 – Benchmark Series.any with skipna=True on boolean Series.

    Parameters:
    - N: Series length
    - TP: True percentage
    """

    def setup(self):
        params = self.params
        N = int(params["N"])
        TP = float(params["TP"])

        # Create boolean Series with specified percentage of True values
        n_true = int(N * TP)
        n_false = N - n_true
        
        values = [True] * n_true + [False] * n_false
        self.rng.shuffle(values)

        series = pd.Series(values, dtype=bool)

        self.args = {"series": series}

    def run(self):
        return self.args["series"].any(skipna=True)
