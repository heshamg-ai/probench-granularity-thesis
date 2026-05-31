import numpy as np
import pandas as pd
from probench.scenarios.scenario_base import PerfScenario


class PR51921Scenario(PerfScenario):
    """
    PR #51921 – Benchmark pd.factorize() for object columns not containing strings
    
    Parameters:
    - N: Array length
    """
    def setup(self):

        params = self.params
        N = int(params["N"])

        arr = self.rng.integers(0, 1000, size=N).astype(object)
        series = pd.Series(arr, dtype="object")
        
        self.args = {"series": series}

    def run(self):
        return pd.factorize(self.args["series"])