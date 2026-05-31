import numpy as np
import pandas as pd
from probench.scenarios.scenario_base import PerfScenario


class PR55131Scenario(PerfScenario):
    """
    PR #55131 — Performance improvement in DataFrame.groupby for pyarrow timestamp and duration dtypes.

    Parameters:
    - L: DataFrame Length
    - G: Group Cardinality
    """
    def setup(self):
        params = self.params
        L = int(params["L"])
        G = int(params["G"])

        df = pd.DataFrame({
            "group": self.rng.integers(0, G, size=L),
            "value": pd.array(np.arange(L), dtype="timestamp[ns][pyarrow]"),
        })

        self.args = {"grouped": df.groupby("group")["value"]} 

    def run(self):
        return self.args["grouped"].max()