import pandas as pd
from probench.scenarios.scenario_base import PerfScenario


class PR57595Scenario(PerfScenario):
    """
    PR #57595 – Benchmark `DataFrameGroupBy.__len__` for multi-key `groupby`.

    Parameters:
    - R : Number of rows
    - C : Number of columns
    - GC : Number of grouping key columns
    """

    def setup(self):
        params = self.params

        R = int(params["R"])          
        C = int(params["C"])         
        GC = int(params["GC"])        
   
        data = {}
        group_cols = []

        for i in range(GC):
            data[f"key{i}"] = self.rng.integers(0, 100, size=R)
            group_cols.append(f"key{i}")

        num_extra_cols = C - GC
        for j in range(num_extra_cols):
            data[f"col{j}"] = self.rng.integers(0, 100, size=R)

        df = pd.DataFrame(data)

        self.args = {
            "df": df,
            "group_cols": group_cols
        }

    def run(self):
        return len(self.args["df"].groupby(self.args["group_cols"]))
        
