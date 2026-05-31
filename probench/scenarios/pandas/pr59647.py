# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2025, Siemens AG

import numpy as np
import pandas as pd
from probench.scenarios.scenario_base import PerfScenario


class PR59647Scenario(PerfScenario):
    """
    PR #59647 – Benchmark CategoricalDtype.update_dtype

    Parameters:
    N: Number of categories in the dtype
    """
    
    def setup(self):
        params = self.params

        n = int(params["N"])

        base_dtype = pd.CategoricalDtype(ordered=False)

        new_dtype = pd.CategoricalDtype(
            categories=list(range(n)),
            ordered=True
        )

        self.args = {
            "base": base_dtype,
            "new": new_dtype
        }

    def run(self):
        return self.args["base"].update_dtype(self.args["new"])



