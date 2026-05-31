import pandas as pd
import pyarrow as pa
import numpy as np
from pandas.core.arrays.arrow.array import ArrowExtensionArray
from probench.scenarios.scenario_base import PerfScenario


class PR51730Scenario(PerfScenario):
    """
    PR #51730 — ArrowExtensionArray._from_sequence_of_strings when parsing strings to boolean[pyarrow] dtype
    
    Parameters:
    - N: Number of strings to parse
    - B: Backend library for string object
    """
    def setup(self):
        params = self.params

        N = int(params["N"])
        B = params["B"]

        if B == "numpy":
            strings = self.rng.choice(["true", "false", None], N)
        elif B == "pyarrow":
            string_np = self.rng.choice(["true", "false", None], N)
            strings = pa.array(string_np, type=pa.string())

        self.args = {"strings" : strings}

    def run(self):
        return ArrowExtensionArray._from_sequence_of_strings(self.args["strings"], dtype=pa.bool_())