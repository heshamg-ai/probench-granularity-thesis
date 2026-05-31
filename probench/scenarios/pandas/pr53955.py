import pandas as pd
from probench.scenarios.scenario_base import PerfScenario


class PR53955Scenario(PerfScenario):
    """
    PR #53955 – Benchmark MultiIndex set and indexing operations.

    Parameters:
    - N: Index size
    - M: Monotonicity (monotonic/non_monotonic)
    - DT: Data type (int/string/datetime)
    """

    def setup(self):
        params = self.params
        N = int(params["N"])
        M = params["M"]
        DT = params["DT"]

        level1_size = max(int(N ** 0.5), 100)
        level2_size = N // level1_size

        if DT == "int":
            level1 = range(level1_size)
            level2 = range(level2_size)
        elif DT == "string":
            level1 = range(level1_size)
            level2 = [f"i-{i}" for i in range(level2_size)]
        else:  # datetime
            level1 = range(level1_size)
            level2 = pd.date_range(start="2000-01-01", periods=level2_size)

        idx1 = pd.MultiIndex.from_product([level1, level2])

        if M == "non_monotonic":
            idx1 = idx1[::-1]

        idx2 = idx1[:-1]

        self.args = {
            "idx1": idx1,
            "idx2": idx2
        }

    def run(self):
        return self.args["idx1"].intersection(self.args["idx2"])