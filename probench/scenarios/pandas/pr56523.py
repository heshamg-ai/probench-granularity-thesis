import numpy as np
import pandas as pd
from probench.scenarios.scenario_base import PerfScenario


class PR56523Scenario(PerfScenario):
    """
    PR #56523 – Benchmark merge with monotonic, sorted keys.

    Parameters:
    - R : Number of rows in each DataFrame
    """

    def setup(self):
        params = self.params
        R = int(params["R"])

        left_keys = np.arange(0, R, 2)
        right_keys = np.arange(R//2, R//2 + R, 1)

        left = pd.DataFrame({"key": left_keys, "val1": 1})
        right = pd.DataFrame({"key": right_keys, "val2": 2})

        self.args = {"left": left,"right": right}

    def run(self):
        return self.args["left"].merge(self.args["right"], on="key", how="inner")
