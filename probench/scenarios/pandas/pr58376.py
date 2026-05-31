import pandas as pd
from probench.scenarios.scenario_base import PerfScenario


class PR58376Scenario(PerfScenario):
    """
    PR #58376 – Benchmark RangeIndex.value_counts()

    Parameters:
    L: RangeIndex Length
    """
   
    def setup(self):
        params = self.params
        L = int(params["L"])

        self.args = {
            "ri": pd.RangeIndex(L),
        }

    def run(self):
        return self.args["ri"].value_counts()