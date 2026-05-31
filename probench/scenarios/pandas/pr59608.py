import pandas as pd
from probench.scenarios.scenario_base import PerfScenario


class PR59608Scenario(PerfScenario):
    """
    PR #59608 – Benchmark DataFrame.to_csv(index=False) for a DataFrame with MultiIndex

    Parameters:
    - R: Number of rows
    - C: Number of columns
    - L: Number of levels in the MultiIndex
    """

    def setup(self):
        params = self.params

        rows = int(params["R"])
        cols = int(params["C"])
        levels = int(params["L"])

        df = pd.DataFrame(
            self.rng.random((rows, cols)),
            columns=[f"col_{i}" for i in range(cols)]
        )

        index_cols = [f"col_{i}" for i in range(levels)]
        df = df.set_index(index_cols, drop=False)

        self.args = {
            "df": df,
            "path" :  "/tmp/export_test.csv" 
        }

    def run(self):
        return self.args["df"].to_csv(self.args["path"], index=False)

