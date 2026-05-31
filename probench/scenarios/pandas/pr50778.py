import numpy as np
import pandas as pd
from probench.scenarios.scenario_base import PerfScenario

class PR50778Scenario(PerfScenario):
    """
    PR #50778 – Benchmark Series.replace when using a large dict for to_replace

    Parameters:
    - N: Series Length
    - R: Number to replace
    """
    def setup(self):
        params = self.params
        N = int(params["N"])
        num_to_replace = int(params["R"])

        # Generate data using self.rng
        arr = self.rng.standard_normal(N)
        arr1 = arr.copy()
        self.rng.shuffle(arr1)

        ser = pd.Series(arr)
        to_replace_list = self.rng.choice(arr, num_to_replace)
        values_list = self.rng.choice(arr1, num_to_replace)

        replace_dict = dict(zip(to_replace_list, values_list))

        self.args = {
            "series": ser,
            "replace_dict": replace_dict
        }

    def run(self):
        return self.args["series"].replace(self.args["replace_dict"])