# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2025, Siemens AG

import pandas as pd
from probench.scenarios.scenario_base import PerfScenario


class PR58817Scenario(PerfScenario):
    """
    PR #58817 – Benchmark DataFrame.unstack() when the DataFrame index is not a MultiIndex

    Parameters:
    - R: Number of rows
    - C: Number of columns
    """

    def setup(self):
        params = self.params

        rows = int(params["R"])
        cols = int(params["C"])

        df = pd.DataFrame(self.rng.random((rows, cols)))

        self.args = {"df": df}

    def run(self):
        return self.args["df"].unstack()

