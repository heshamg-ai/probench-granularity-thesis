import math
from pandas import MultiIndex, date_range
from probench.scenarios.scenario_base import PerfScenario


class PR55108Scenario(PerfScenario):
    """
    PR #55108 — Performance improvement in Index.difference.

    Parameters:
    - L1: base index length
    - SL: subset length
    """

    def setup(self):
        L1 = int(self.params["L1"])
        SL = min(int(self.params["SL"]), L1)

        n_level1 = 1000
        n_level2 = max(1, math.ceil(L1 / n_level1))

        level1 = range(n_level1)
        level2 = date_range("2000-01-01", periods=n_level2, freq="D")

        left = MultiIndex.from_product([level1, level2])[:L1]
        right = left[:SL]

        self.args = {
            "base_index": left,
            "other": right,
        }

    def run(self):
        return self.args["base_index"].difference(self.args["other"])