import numpy as np
import pandas as pd
from probench.scenarios.scenario_base import PerfScenario


class PR51873Scenario(PerfScenario):
    """
    PR #51873 – Benchmark MultiIndex.set_levels when replacing a single level.

    Parameters:
    - N: Number of rows in the MultiIndex
    - L: Number of levels in the MultiIndex
    """

    def setup(self):
        params = self.params
        N = int(params["N"])
        L = int(params["L"])

        arrays = [self.rng.integers(0, 100, size=N) for _ in range(L)]
        names = [f"level_{i}" for i in range(L)]
        mi = pd.MultiIndex.from_arrays(arrays, names=names)
        new_level = pd.Index(np.arange(100))

        self.args = {"mi": mi, "new_level": new_level}

    def run(self):
        return self.args["mi"].set_levels(self.args["new_level"], level=0)
