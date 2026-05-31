import pandas as pd
from probench.scenarios.scenario_base import PerfScenario

import numpy as np

class PR56841Scenario(PerfScenario):
    """
    PR #56841 – Benchmark Index.join for left join with many:1 (right unique), right join with 1:many (left unique)

    Parameters:
    - N: Index Size
    - RF: Repeat Factor
    """

    def setup(self):
        params = self.params
        N = int(params["N"])
        RF = int(params["RF"])   

        base = np.arange(N)

        left = pd.Index(base).repeat(RF)
        right = pd.Index(base)

        self.args = {"left": left, "right": right}

    def run(self):
        return self.args["left"].join(self.args["right"], how="left")
