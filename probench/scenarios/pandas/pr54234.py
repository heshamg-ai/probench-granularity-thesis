import pandas as pd
from probench.scenarios.scenario_base import PerfScenario


class PR54234Scenario(PerfScenario):
    """
    PR #54234 – Benchmark DataFrameGroupBy.idxmax / SeriesGroupBy.idxmax.

    Parameters:
    - N : Number of rows
    - G : Number of groups
    """

    def setup(self):
        params = self.params
        N = int(params["N"])
        G = int(params["G"])

        groups = self.rng.integers(0, G, size=N)
        values = self.rng.standard_normal(N)

        df = pd.DataFrame({"group": groups, "value": values})

        self.args = {"df": df}

    def run(self):
        return self.args["df"].groupby("group").idxmax()
