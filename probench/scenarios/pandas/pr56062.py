import pandas as pd
from probench.scenarios.scenario_base import PerfScenario


class PR56062Scenario(PerfScenario):
    """
    PR #56062 – Benchmark DataFrame.loc with a MultiIndex as the key.

    Parameters:
    - N : Number of unique values in the first MultiIndex level
    """

    def setup(self):
        params = self.params
        N = int(params["N"])

        idx = pd.MultiIndex.from_product(
            [range(N), range(10)], names=["a", "b"]
        )
        df = pd.DataFrame({"value": range(len(idx))}, index=idx)
        key = df.index[::10]

        self.args = {"df": df, "key": key}

    def run(self):
        return self.args["df"].loc[self.args["key"]]
