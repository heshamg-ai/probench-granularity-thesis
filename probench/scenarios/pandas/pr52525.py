import numpy as np
import pandas as pd
import pyarrow as pa
from probench.scenarios.scenario_base import PerfScenario


class PR52525Scenario(PerfScenario):
    """
    PR #52525 – Benchmark ArrowExtensionArray.to_numpy with distinct scenarios.
    Tests three scenarios: no missing, some missing, null[pyarrow] type.

    Parameters:
    - N: Array size
    - MP: Missing pattern (none/some/all)
    """

    def setup(self):
        params = self.params

        N = int(params["N"])
        MP = params["MP"]
        
        arrow_type = pa.float64()
        values = list(self.rng.random(N))

        # Apply missing pattern
        if MP == "none":
            arr = pa.array(values, type=arrow_type)
        elif MP == "some":
            n_missing = N // 10
            missing_indices = self.rng.choice(N, size=n_missing, replace=False)
            for idx in missing_indices:
                values[idx] = None
            arr = pa.array(values, type=arrow_type)
        elif MP == "all":
            arr = pa.array([None] * N, type=arrow_type)

        ext_arr = pd.arrays.ArrowExtensionArray(arr)
        self.args = {"ext_arr": ext_arr}

    def run(self):
        return self.args["ext_arr"].to_numpy(dtype="float64", na_value=np.nan)