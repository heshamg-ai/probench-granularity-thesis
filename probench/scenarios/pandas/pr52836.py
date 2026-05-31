import numpy as np
import pandas as pd
from probench.scenarios.scenario_base import PerfScenario


class PR52836Scenario(PerfScenario):
    """
    PR #52836 – Benchmark DataFrame.transpose with masked (nullable Int64) arrays.

    Parameters:
    - R: Number of rows.
    - C: Number of columns.
    - NAP: Fraction of rows set to missing values (pd.NA) in each column.
    """

    def setup(self):
        params = self.params
        R = int(params["R"])
        C = int(params["C"])
        NAP = float(params.get("NAP"))

        n_na = int(R * NAP)

        data = {}
        for i in range(C):
            arr = pd.array(self.rng.integers(0, 1000, size=R), dtype="Int64")

            if n_na > 0:
                na_indices = self.rng.choice(R, size=min(n_na, R), replace=False)
                arr[na_indices] = pd.NA

            data[f"col_{i}"] = arr

        self.args = {"df": pd.DataFrame(data)}

    def run(self):
        return self.args["df"].transpose()

