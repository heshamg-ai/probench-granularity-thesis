import numpy as np
import pandas as pd
from probench.scenarios.scenario_base import PerfScenario


class PR52998Scenario(PerfScenario):
    """
    PR #52998 – Benchmark array.dtype property on IntegerArray.

    Parameters:
    - N: Array size
    - DT: Dtype (Int64/Int32/Int16/Int8)
    """

    def setup(self):
        params = self.params

        N = int(params["N"])
        DT = params["DT"]

        arr = pd.array(self.rng.integers(0, 100, size=N), dtype=DT)
        self.args = {"arr": arr}

    def run(self):
        return self.args["arr"].dtype
