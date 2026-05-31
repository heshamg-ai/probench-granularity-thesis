import numpy as np
import pandas as pd
import pyarrow as pa
from probench.scenarios.scenario_base import PerfScenario


class PR51574Scenario(PerfScenario):
    """
    PR #51574 – Benchmark DataFrame.where with EA-backed DataFrames.

    Parameters:
    - R: Number of rows
    - C: Number of columns
    - DT: Extension Array dtype (Float64/Int64/float64_pyarrow)
    """

    def setup(self):
        params = self.params
        R = int(params["R"])
        C = int(params["C"])
        DT = params["DT"]

        # Create DataFrame with Extension Array dtype
        data = {}
        for i in range(C):
            if DT == "Float64":
                data[f"col_{i}"] = pd.Series(self.rng.random(R), dtype="Float64")
            elif DT == "Int64":
                data[f"col_{i}"] = pd.Series(self.rng.integers(0, 1000, size=R), dtype="Int64")
            elif DT == "float64_pyarrow":
                data[f"col_{i}"] = pd.Series(self.rng.random(R), dtype=pd.ArrowDtype(pa.float64()))

        df = pd.DataFrame(data)

        # Create condition mask as DataFrame (also EA-backed)
        cond = df > 0.5
        self.args = {"df": df, "cond": cond}

    def run(self):
        return self.args["df"].where(self.args["cond"])
