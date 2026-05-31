import numpy as np
import pandas as pd
from probench.scenarios.scenario_base import PerfScenario


class PR54508Scenario(PerfScenario):
    """
    PR #54508 — DataFrame.iloc[int] for EA dtypes.

    Parameters:
    - N: Number of rows
    - C: Number of columns
    - DT: Extension Array Dtype
    """
    def setup(self):
        params = self.params
        R = int(params["R"])
        C = int(params["C"])
        dtype = params["DT"]
        
        data = self.rng.random((R, C))
        df = pd.DataFrame(data, dtype=dtype)

        self.args = {"df": df}

    def run(self):
        return self.args["df"].iloc[1] # 1 is dummy indexer, it dont have effect on performance