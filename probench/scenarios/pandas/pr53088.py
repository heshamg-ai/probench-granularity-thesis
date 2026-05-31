import numpy as np
import pandas as pd
from probench.scenarios.scenario_base import PerfScenario


class PR53088Scenario(PerfScenario):
    """
    PR #53088 – Benchmark GroupBy.groups on large DataFrames.

    Parameters:
    - R: Number of rows
    - G: Number of groups
    - NGC: Number of grouping columns
    """

    def setup(self):
        params = self.params

        R = int(params["R"])
        G = int(params["G"])
        NGC = 2

        data = {}
        for i in range(NGC):
            data[f'col{i}'] = self.rng.integers(0, G, size=R)
        
        data['value'] = self.rng.random(R)

        df = pd.DataFrame(data)
        groupby_cols = [f'col{i}' for i in range(NGC)]

        self.args = {"df": df, "groupby_cols": groupby_cols}

    def run(self):
        return self.args["df"].groupby(self.args["groupby_cols"]).groups
