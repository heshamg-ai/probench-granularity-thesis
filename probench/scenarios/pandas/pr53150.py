import numpy as np
import pandas as pd
import pyarrow as pa
from probench.scenarios.scenario_base import PerfScenario


class PR53150Scenario(PerfScenario):
    """
    PR #53150 – Benchmark Series.add with Arrow string/binary types.

    Parameters:
    - N: Series length
    - DT: Dtype (string_pyarrow/binary_pyarrow)
    - OT: Operand type (series/scalar)
    """

    def setup(self):
        params = self.params

        N = int(params["N"])
        DT = params["DT"]
        OT = params["OT"]

        # Create Series with Arrow dtype
        if DT == "string_pyarrow":
            s1 = pd.Series([f"str_{i}" for i in range(N)], dtype=pd.ArrowDtype(pa.string()))
            if OT == "series":
                s2 = pd.Series([f"_end_{i}" for i in range(N)], dtype=pd.ArrowDtype(pa.string()))
            else:  # scalar
                s2 = "_suffix"
        elif DT == "binary_pyarrow":
            s1 = pd.Series([f"bin_{i}".encode() for i in range(N)], dtype=pd.ArrowDtype(pa.binary()))
            if OT == "series":
                s2 = pd.Series([f"_end_{i}".encode() for i in range(N)], dtype=pd.ArrowDtype(pa.binary()))
            else:  # scalar
                s2 = b"_suffix"

        self.args = {"s1": s1, "s2": s2}

    def run(self):
        return self.args["s1"].add(self.args["s2"])
