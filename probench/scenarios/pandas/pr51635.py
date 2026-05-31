import numpy as np
import pandas as pd
import pyarrow as pa
from probench.scenarios.scenario_base import PerfScenario


class PR51635Scenario(PerfScenario):
    """
    PR #51635 – Benchmark ArrowExtensionArray.fillna with no missing values.

    Parameters:
    - N: Array size
    - AT: Arrow type (int64/float64/string)
    """

    def setup(self):
        params = self.params
        N = int(params["N"])
        AT = params["AT"]

        # Create Arrow array based on type
        if AT == "int64":
            arrow_type = pa.int64()
            values = list(self.rng.integers(0, 1000, size=N))
            fill_value = 0  # valid for int
        elif AT == "float64":
            arrow_type = pa.float64()
            values = list(self.rng.random(N))
            fill_value = 0.0  # valid for float
        elif AT == "string":
            arrow_type = pa.string()
            values = [f"str_{i}" for i in range(N)]
            fill_value = ""  # valid for string (empty string is fine)

        arr = pa.array(values, type=arrow_type)
        ext_arr = pd.arrays.ArrowExtensionArray(arr)

        self.args = {"ext_arr": ext_arr, "fill_value": fill_value}

    def run(self):
        return self.args["ext_arr"].fillna(self.args["fill_value"])
