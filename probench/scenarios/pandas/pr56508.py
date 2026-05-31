import pandas as pd
from probench.scenarios.scenario_base import PerfScenario


class PR56508Scenario(PerfScenario):
    """
    PR #56508 – Benchmark pandas.util.hash_array with NA values.

    Parameters:
    - N: Array size
    - NAP: NA percentage
    - VT: Value type (int/float/boolean) - must be masked array types that inherit from BaseMaskedArray
    """

    def setup(self):
        params = self.params
        N = int(params["N"])
        NAP = float(params["NAP"])
        VT = params["VT"]

        # Create base data based on value type
        # NOTE: Only masked array types (Int64, Float64, boolean) benefit from PR #56508
        # StringArray does NOT inherit from BaseMaskedArray and won't use the optimization
        if VT == "int":
            data = self.rng.integers(0, 1000, size=N)
            arr = pd.array(data, dtype="Int64")
        elif VT == "float":
            data = self.rng.random(N)
            arr = pd.array(data, dtype="Float64")
        elif VT == "boolean":
            data = self.rng.choice([True, False], size=N)
            arr = pd.array(data, dtype="boolean")

        # Add NA values
        if NAP > 0:
            n_na = int(N * NAP)
            na_indices = self.rng.choice(N, size=n_na, replace=False)
            arr[na_indices] = pd.NA
        
        self.args = {"array": arr}

    def run(self):
        return pd.util.hash_array(self.args["array"])