from probench.scenarios.scenario_base import PerfScenario

import pandas as pd

class PR57787Scenario(PerfScenario):
    """
    PR #57787 – Benchmark pd.Categorical.categories when values is a range

    Parameters:
    N : Number of elements for Categorical object 
    """
    def setup(self):
        params = self.params

        N = int(params["N"])
        values = range(N)

        self.args = {
            "values": values
        }

    def run(self):
        return pd.Categorical(values=self.args["values"], categories=None).categories

