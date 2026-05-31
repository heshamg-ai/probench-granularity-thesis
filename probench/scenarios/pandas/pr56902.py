import pandas as pd
from probench.scenarios.scenario_base import PerfScenario
import numpy as np

class PR56902Scenario(PerfScenario):
    """
    PR #56902 – Benchmark `DataFrameGroupBy.ffill` with NaNs across multiple groups.

    Parameters:
    - R : Rows per group.
    - G : Number of groups.
    """

    def setup(self):
        params = self.params
        R = int(params["R"])      # rows per group
        G = int(params["G"])      # number of groups

        total = R * G
        groups = np.repeat(np.arange(G), R)
        values = np.tile([np.nan, 1.0], total // 2 + 1)[:total]

        df = pd.DataFrame({"group": groups,
                           "value": values}).set_index("group")

        df = df.sample(frac=1.0, random_state=self.seed)
        self.args = {"df": df}

    def run(self):
        return self.args["df"].groupby("group").ffill()




