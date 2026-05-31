# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2025, Siemens AG

import pandas as pd
import numpy as np
from probench.scenarios.scenario_base import PerfScenario

class PR51777Scenario(PerfScenario):
    """
    PR #51777 – Benchmark Series.combine_first for misaligned indexes.

    Parameters:
    - N: Series length
    - D: Shift size (drop count). s1 drops the last D rows, s2 drops the first D rows,
         creating misalignment with partial overlap.
    """

    def setup(self):
        N = 5000000
        drop = int(self.params["D"])

        s1 = pd.Series(self.rng.integers(0, N, N), dtype="int64").iloc[:-drop]
        s2 = pd.Series(self.rng.integers(0, N, N), dtype="int64").iloc[drop:]

        self.args = {"s1": s1, "s2": s2}

    def run(self):
        return self.args["s1"].combine_first(self.args["s2"])



