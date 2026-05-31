import pandas as pd
from probench.scenarios.scenario_base import PerfScenario
import numpy as np

class PR58018Scenario(PerfScenario):
    """
    PR #58018 – Benchmark Index.to_frame()

    Parameters:
    L: RangeIndex Length
    """

    def setup(self):
        params = self.params

        L = int(params["L"])
        idx = pd.Index(np.arange(L), name=None)

        self.args = {"idx": idx}

    def run(self):
        return self.args["idx"].to_frame()

