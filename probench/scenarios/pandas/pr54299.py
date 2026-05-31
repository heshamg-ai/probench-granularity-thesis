import numpy as np
import pandas as pd
import json
from probench.scenarios.scenario_base import PerfScenario


class PR54299Scenario(PerfScenario):
    """
    PR #54299 — Performance improvement in DataFrame.astype for ExtensionDtype.
    
    Parameters:
    - R: Number of rows
    - C: Number of columns
    - BaseDT: Dtype pairs that specifies the conversion direction
    """
    def setup(self):
        params = self.params
        R = int(params["R"])
        C = int(params["C"])
        base_dtype, to_dtype = json.loads(params["BaseDT"])

        data = np.random.randint(0, 100, (R, C))
        df = pd.DataFrame(data, dtype=base_dtype)
    
        self.args = {
            "df": df,
            "to_dtype": to_dtype
        }

    def run(self):
        self.args["df"].astype(self.args["to_dtype"], copy=False)