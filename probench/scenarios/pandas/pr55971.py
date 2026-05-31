import numpy as np
import pandas as pd
from probench.scenarios.scenario_base import PerfScenario


class PR55971Scenario(PerfScenario):
    """
    PR #55971 – Benchmark pd.testing.assert_series_equal for identical Series.

    Parameters:
    - N: Number of elements in each Series
    """

    def setup(self):
        params = self.params
        N = int(params["N"])

        chars = np.array(list("ABCDEFGHIJKLMNOPQRSTUVWXYZ"), dtype="U1")
        idx = pd.Index(
            ["".join(row) for row in chars[self.rng.integers(0, 26, size=(N, 10))]]
        )
        ser1 = pd.Series(1, index=idx.copy(deep=True))
        ser2 = pd.Series(1, index=idx.copy(deep=True))

        self.args = {"ser1": ser1, "ser2": ser2}

    def run(self):
        return pd.testing.assert_series_equal(self.args["ser1"], self.args["ser2"])
