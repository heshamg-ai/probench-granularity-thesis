import numpy as np
import pandas as pd
import pyarrow as pa
from probench.scenarios.scenario_base import PerfScenario

class PR51347Scenario(PerfScenario):
    """
    PR #51347 – Benchmark ArrowExtensionArray.to_numpy(dtype=object) when nulls are present
    
    Parameters:
    - N: Array size
    - NP: Null percentage
    """

    def setup(self):
        params = self.params
        N = int(params["N"])
        NP = float(params["NP"])

        values = self.rng.random(N).tolist()

        if NP > 0:
            n_nulls = int(N * NP)
            null_indices = self.rng.choice(N, size=n_nulls, replace=False)
            for idx in null_indices:
                values[idx] = None

        # Create ArrowExtensionArray
        arr = pa.array(values, type=pa.float64())
        ext_arr = pd.arrays.ArrowExtensionArray(arr)

        self.args = {"ext_arr":ext_arr}

    def run(self):
        return self.args["ext_arr"].to_numpy(dtype=object)
