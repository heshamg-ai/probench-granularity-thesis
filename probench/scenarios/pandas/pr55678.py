import pandas as pd
from probench.scenarios.scenario_base import PerfScenario


class PR55678Scenario(PerfScenario):
    """
    PR #55678 – Benchmark `pandas.merge_asof` for "by" is int or object

    Parameters:
    R : Number of rows
    DT : Data type by key columns
    """

    def setup(self):
        R = int(self.params["R"])
        DT = self.params["DT"]

        df1 = pd.DataFrame({
            "time": self.rng.integers(0, R / 20, R),
            "key": self.rng.integers(0, 100, R),
            "key2": self.rng.integers(0, 50, R),
            "v1": self.rng.standard_normal(R),
        })

        df2 = pd.DataFrame({
            "time": self.rng.integers(0, R / 20, R),
            "key": self.rng.integers(0, 100, R),
            "key2": self.rng.integers(0, 50, R),
            "v2": self.rng.standard_normal(R),
        })

        if DT == "obj":
            df1["key"] = df1["key"].astype(str)
            df2["key"] = df2["key"].astype(str)

        df1 = df1.sort_values("time")
        df2 = df2.sort_values("time")

        self.args = {
            "left": df1[["time", "key", "key2", "v1"]],
            "right": df2[["time", "key", "key2", "v2"]],
            "on": "time",
            "by": ["key"]
        }

    def run(self):
        return pd.merge_asof(**self.args)

