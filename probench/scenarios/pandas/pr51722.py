import pandas as pd
from probench.scenarios.scenario_base import PerfScenario


class PR51722Scenario(PerfScenario):
    """
    PR #51722 – Benchmark GroupBy.quantile with many groups.

    Parameters:
    - R: Number of rows
    - G: Number of groups
    - C: Number of columns
    """

    def setup(self):
        R = int(self.params["R"])
        G = int(self.params["G"])
        C = int(self.params["C"])

        # Create numeric data using your RNG
        arr = self.rng.standard_normal((R, C))
        df = pd.DataFrame(arr, columns=[f"col_{i}" for i in range(C)])

        # Create group column
        df["group"] = self.rng.integers(0, G, size=R)
        grouped_df = df.groupby("group")

        self.args = {"grouped_df": grouped_df}

    def run(self):
        return self.args["grouped_df"].quantile([0.5, 0.75])
