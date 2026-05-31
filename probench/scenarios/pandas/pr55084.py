import numpy as np
import pandas as pd
from probench.scenarios.scenario_base import PerfScenario


class PR55084Scenario(PerfScenario):
    """
    PR #55084 — concat(axis=1) with unaligned indexes.

    Parameters:
    - L: Series Length
    - C: Number of Series
    - O: Index Overlap Percentage
    """
    def setup(self):
        params = self.params
        L = int(params["L"])
        C = int(params["C"])
        O = float(params["O"])

        overlap_size = int(L * O)
        step_size = L - overlap_size
        
        series_list = []
        for i in range(C):
            start = i * step_size
            end = start + L
            index = pd.Index(np.arange(start, end))
            series = pd.Series(self.rng.random(L), index=index)
            series_list.append(series)

        self.args = {"series_list": series_list}

    def run(self):
        return pd.concat(self.args["series_list"], axis=1)