# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2025, Siemens AG

import numpy as np
import pandas as pd
from probench.scenarios.scenario_base import PerfScenario


class PR61116Scenario(PerfScenario):
    """
    PR #61116 – Benchmark Series.nlargest() with duplicates

    Parameters
    N : Length of the Series
    DP : Duplicate Percentage
    """

    def setup(self):
        params = self.params
        N = int(params["N"])
        DP = float(params["DP"])

        values = self.rng.random(N)

        k = int(N * DP)
        n_unique = N - k

        unique_index = np.arange(n_unique)
        duplicate_index = self.rng.choice(unique_index, size=k, replace=True)

        index = np.concatenate([unique_index, duplicate_index])
        self.rng.shuffle(index)

        self.args = {
            "series": pd.Series(values, index=index)
        }

    def run(self):
        return self.args["series"].nlargest(100)


