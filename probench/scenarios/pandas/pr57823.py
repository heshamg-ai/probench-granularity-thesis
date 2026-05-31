import pandas as pd 
from probench.scenarios.scenario_base import PerfScenario

class PR57823Scenario(PerfScenario):
    """
    PR #57823 – Benchmark RangeIndex.argmin()

    Parameters:
    L: RangeIndex Length
    """

    def setup(self):
        params = self.params
        L = int(params["L"])       
        ri = pd.RangeIndex(L)

        self.args = {
            "ri": ri
        }

    def run(self):
        return self.args["ri"].argmin()
