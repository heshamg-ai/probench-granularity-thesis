import numpy as np
import pandas as pd
import pyarrow as pa
from probench.scenarios.scenario_base import PerfScenario


class PR51630Scenario(PerfScenario):
    """
    PR #51630 – Benchmark ArrowExtensionArray.isna on DataFrames.

    Parameters:
    - R: Number of rows
    - C: Number of columns
    - AT: Arrow type (int64/float64/string)
    - NP: Null pattern (no_nulls/all_nulls)
    """

    def setup(self):
        params = self.params
        R = int(params["R"])
        C = int(params["C"])
        AT = params["AT"]
        NP = params["NP"]

        # Create DataFrame with Arrow columns
        data = {}
        
        for i in range(C):
            # Create base values
            if AT == "int64":
                arrow_type = pa.int64()
                values = list(self.rng.integers(0, 1000, size=R))
            elif AT == "float64":
                arrow_type = pa.float64()
                values = list(self.rng.random(R))
            elif AT == "string":
                arrow_type = pa.string()
                values = [f"str_{j}" for j in range(R)]

            # Apply null pattern
            if NP == "no_nulls":
                # No nulls (fast path edge case)
                arr = pa.array(values, type=arrow_type)
            elif NP == "all_nulls":
                # Entirely null (fast path edge case)
                arr = pa.array([None] * R, type=arrow_type)

            data[f"col_{i}"] = pd.arrays.ArrowExtensionArray(arr)

        df = pd.DataFrame(data)
        self.args = {"df": df}

    def run(self):
        return self.args["df"].isna()
