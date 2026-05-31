import pandas as pd
from probench.scenarios.scenario_base import PerfScenario

class PR56990Scenario(PerfScenario):

    def setup(self):
        """
        PR #56990 – Benchmark `MultiIndex.equals` on a MultiIndex.

        Parameters
        L : Length of each level 
        """
        params = self.params

        L = int(params["L"])     
        mi = pd.MultiIndex.from_product(
            [
                pd.date_range("2000-01-01", periods=L),
                pd.RangeIndex(L),
            ]
        )
        mi_copy = mi.copy(deep=True)

        self.args = {"mi": mi, "mi_copy": mi_copy}

    def run(self):
        return self.args["mi"].equals(self.args["mi_copy"])
