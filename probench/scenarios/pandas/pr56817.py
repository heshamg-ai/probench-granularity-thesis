import numpy as np
import pandas as pd
from probench.scenarios.scenario_base import PerfScenario


class PR56817Scenario(PerfScenario):
    """
    PR #56817 – Benchmark Index.join when left and/or right are non-unique

    Parameters:
    - N: Index Size
    - TF: Tile factor
    """

    def setup(self):
        params = self.params
        N = int(params["N"])
        TF = 100

        left = pd.Index(np.tile(np.arange(N), TF))          # non-unique, not sorted
        right = pd.Index(np.arange(max(1, N // 10)))        # smaller unique right

        self.args = {"left": left, "right": right}

    def run(self):
        return self.args["left"].join(self.args["right"], how="left")