import pandas as pd
import pyarrow as pa
from probench.scenarios.scenario_base import PerfScenario


class PR54224Scenario(PerfScenario):
    """
    PR #54224 – Benchmark DataFrame.transpose with pyarrow arrays.

    Parameters:
    - R: Number of rows
    - C: Number of columns
    - DT: Data type (float64_pyarrow/int64_pyarrow/string_pyarrow)
    """

    def setup(self):
        params = self.params
        R = int(params["R"])
        C = int(params["C"])
        DT = params["DT"]

        # Create data based on dtype
        if DT == "float64_pyarrow":
            data = {f"col_{i}": pd.array(self.rng.random(R), dtype=pd.ArrowDtype(pa.float64())) for i in range(C)}
        elif DT == "int64_pyarrow":
            data = {f"col_{i}": pd.array(self.rng.integers(0, 1000, size=R), dtype=pd.ArrowDtype(pa.int64())) for i in range(C)}
        elif DT == "string_pyarrow":
            data = {f"col_{i}": pd.array([f"str_{j}" for j in range(R)], dtype=pd.ArrowDtype(pa.string())) for i in range(C)}

        df = pd.DataFrame(data)

        self.args = {"df": df}

    def run(self):
        return self.args["df"].transpose()
