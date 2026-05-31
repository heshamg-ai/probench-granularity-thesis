import numpy as np
import pandas as pd
from probench.scenarios.scenario_base import PerfScenario


class PR53013Scenario(PerfScenario):
    """
    PR #53013 – Benchmark array.reshape(-1, 1) on IntegerArray.

    Parameters:
    - N: Array size
    - DT: Dtype (Int64/Int32)
    """

    def setup(self):
        params = self.params

        N = int(params["N"])
        DT = params["DT"]

        # Create IntegerArray or FloatArray
        if DT == "Int64":
            arr = pd.array(self.rng.integers(0, 1000, size=N), dtype="Int64")
        elif DT == "Int32":
            arr = pd.array(self.rng.integers(0, 1000, size=N), dtype="Int32")

        self.args = {"arr": arr}

    def run(self):
        return self.args["arr"].reshape(-1, 1)
