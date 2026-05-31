# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2025, Siemens AG

import pandas as pd
from probench.scenarios.scenario_base import PerfScenario


class PR61014Scenario(PerfScenario):
    """
    PR 61014 when ``key`` is a DataFrame with many columns
    
    Parameters:
    - R: Number of rows
    - C: Number of columns
    """

    def setup(self):
        params = self.params

        rows = 10
        cols = int(params["C"])

        df = pd.DataFrame(self.rng.standard_normal((rows, cols)))
        mask = df > 0.5

        self.args = {
            "df": df,
            "mask": mask
        }

    def run(self):
        return self.args["df"].where(self.args["mask"])


