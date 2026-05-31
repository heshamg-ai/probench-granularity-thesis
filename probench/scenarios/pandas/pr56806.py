import numpy as np
import pandas as pd
from probench.scenarios.scenario_base import PerfScenario


class PR56806Scenario(PerfScenario):
    """
    PR #56806 – Benchmark Index.take to check for full range indices

    Parameters:
    - N: Index size
    """

    def setup(self):
        params = self.params
        N = int(params["N"])
       
        idx = pd.Index(np.arange(N))
        indices = np.arange(N)
    
        self.args = {"idx":idx, "indices": indices}

    def run(self):
        return self.args["idx"].take(self.args["indices"])
