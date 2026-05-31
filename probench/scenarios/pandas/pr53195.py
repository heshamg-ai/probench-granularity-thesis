import pandas as pd
from probench.scenarios.scenario_base import PerfScenario



class PR53195Scenario(PerfScenario):
    """
    PR #53195 – Benchmark GroupBy.apply for group_keys=True with many groups

    Parameters:
    - R: Number of rows
    - G: Number of groups (PR specifically optimizes for many groups)
    """

    def setup(self):
        params = self.params
        R = int(params["R"])
        G = int(params["G"])

        df = pd.DataFrame({
            'group': self.rng.integers(0, G, size=R),
            'value': self.rng.random(R)
        })
        
        self.args = {"df": df}

    def run(self):
        return self.args["df"].groupby('group', group_keys=True).apply(lambda x: x)
