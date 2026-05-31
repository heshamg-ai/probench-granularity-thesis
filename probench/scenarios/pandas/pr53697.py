import pandas as pd
from probench.scenarios.scenario_base import PerfScenario


class PR53697Scenario(PerfScenario):
    """
    PR #53697 – Benchmark MultiIndex.append with large indices.

    Parameters:
    - N: Index size
    - L: Number of levels
    """

    def setup(self):
        params = self.params
        N = int(params["N"])
        L = int(params["L"])

        arrays1 = [self.rng.integers(0, 100, size=N) for _ in range(L)]
        names = [f"level_{i}" for i in range(L)]
        idx1 = pd.MultiIndex.from_arrays(arrays1, names=names)

        arrays2 = [self.rng.integers(0, 100, size=N) for _ in range(L)]
        idx2 = pd.MultiIndex.from_arrays(arrays2, names=names)
       

        self.args = {
            "idx1": idx1,
            "idx2": idx2
        }

    def run(self):
        return self.args["idx1"].append(self.args["idx2"])