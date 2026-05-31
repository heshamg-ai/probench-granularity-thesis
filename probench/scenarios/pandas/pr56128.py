import numpy as np
import pandas as pd
from probench.scenarios.scenario_base import PerfScenario


class PR56128Scenario(PerfScenario):
    """
    PR #56128 – Benchmark Index.sort_values when index is already sorted.

    Parameters:
    - N: Index size
    """

    def setup(self):
        params = self.params
        N = int(params["N"])

        idx = pd.Index(np.arange(N)) #monotonic increasing
    
        self.args = {"idx": idx}

    def run(self):
        return self.args["idx"].sort_values()
