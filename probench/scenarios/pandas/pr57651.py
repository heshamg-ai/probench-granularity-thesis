import pandas as pd
from probench.scenarios.scenario_base import PerfScenario


class PR57651Scenario(PerfScenario):
    """
    PR #57651 – Benchmark RangeIndex.join with overlapping ranges.

    Parameters:
    - L : Length of left RangeIndex
    - R : Length of right RangeIndex
    """

    def setup(self):
        params = self.params
        L = int(params["L"])
        R = int(params["R"])

        left = pd.RangeIndex(0, L)
        right = pd.RangeIndex(L // 2, L // 2 + R)

        self.args = {
            "left": left,
            "right": right,
        }

    def run(self):
        return self.args["left"].join(self.args["right"], how="inner")
