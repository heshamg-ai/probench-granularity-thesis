import numpy as np
import pandas as pd
import pyarrow as pa
from probench.scenarios.scenario_base import PerfScenario


class PR51549Scenario(PerfScenario):
    """
    PR #51549 – Benchmark DataFrame.first_valid_index

    Parameters:
    - R: Number of rows
    - C: Number of columns
    - DT: Extension Array dtype (Float64/float64_pyarrow)
    - NA: Number of leading missing rows (None/pd.NA) injected at the top of each column
    """

    def setup(self):
        params = self.params
        R = int(params["R"])
        C = int(params["C"])
        DT = params["DT"]
        NA_head = int(params["NA"])

        # Create DataFrame with Extension Array dtype
        data = {}
        for i in range(C):
            if DT == "Float64":
                col = pd.Series(self.rng.random(R), dtype="Float64")
            elif DT == "float64_pyarrow":
                col = pd.Series(self.rng.random(R), dtype=pd.ArrowDtype(pa.float64()))

            if NA_head > 0:
                col.iloc[:min(R, NA_head)] = None  # inject leading NAs
            
            data[f"col_{i}"] = col

        df = pd.DataFrame(data)
        self.args = {"df": df}

    def run(self):
        return self.args["df"].first_valid_index()