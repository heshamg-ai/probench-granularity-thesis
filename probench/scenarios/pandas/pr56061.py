import numpy as np
import pandas as pd
from probench.scenarios.scenario_base import PerfScenario


class PR56061Scenario(PerfScenario):
    """
    PR #56061 – Benchmark `DataFrameGroupBy.nunique` with large group cardinality.

    Parameters:
    - R: Total rows
    - C: Cardinality, Number of unique categories
    """

    def setup(self):

        params = self.params
        R = int(params["R"])
        C = int(params["C"])

        # Create group keys with specified cardinality
        group_keys = self.rng.integers(0, C, size=R ,dtype=np.int64)
        values = self.rng.integers(0, 1000, size=R)
       
        # Create DataFrame
        df = pd.DataFrame({
            'group': group_keys,
            'value': values
        })

        self.args = {"df": df}

    def run(self):
        return self.args["df"].groupby('group').nunique()
