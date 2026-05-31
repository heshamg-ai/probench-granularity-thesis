import numpy as np
import pandas as pd
from probench.scenarios.scenario_base import PerfScenario


class PR55580Scenario(PerfScenario):
    """
    PR #55580 – Benchmark `pandas.merge_asof` with multiple `by` keys

    Parameters:
    R : Number of rows
    K : Number of `by` key columns
    """
    def setup(self):
        R = int(self.params["R"])
        K = int(self.params["K"])

        data1 = {
            "time": self.rng.integers(0, R / 20, R),
            "v1": self.rng.standard_normal(R),
        }
        data2 = {
            "time": self.rng.integers(0, R / 20, R),
            "v2": self.rng.standard_normal(R),
        }

        by_cols = []
        
        for i in range(K):
            col = f"key{i}"
            by_cols.append(col)
            data1[col] = self.rng.integers(0, 100, R)
            data2[col] = self.rng.integers(0, 100, R)

        df1 = pd.DataFrame(data1).sort_values("time")
        df2 = pd.DataFrame(data2).sort_values("time")

        self.args = {
            "left": df1[["time", *by_cols, "v1"]],
            "right": df2[["time", *by_cols, "v2"]],
            "on": "time",
            "by": by_cols
        }

    def run(self):
        return pd.merge_asof(**self.args)
