import pandas as pd
from probench.scenarios.scenario_base import PerfScenario


class PR53397Scenario(PerfScenario):
    """
    PR #53397 – Benchmark RangeIndex.take optimization.

    Parameters:
    - N: Index size
    - I: Indices to take
    """

    def setup(self):
        params = self.params
        N = int(params["N"])
        I = int(params["I"])

        idx = pd.RangeIndex(N)
        indices = self.rng.integers(0, N, I)

        self.args = {
            "idx": idx,
            "indices": indices
        }

    def run(self):
        return self.args["idx"].take(self.args["indices"])
