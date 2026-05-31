import pandas as pd
from probench.scenarios.scenario_base import PerfScenario


class PR57023Scenario(PerfScenario):
    """
    PR #57023 – Benchmark Index.join cached-attribute propagation when the result matches an input.

    Parameters
    N : Index Length
    """

    def setup(self):
        params = self.params

        N = int(params["N"])

        data = [f"key-{i:08d}" for i in range(N)]

        idx1 = pd.Index(data, dtype="string[pyarrow_numpy]")
        idx2 = pd.Index(data[1:], dtype="string[pyarrow_numpy]")

        # Pre-warm the cache on idx1 so the optimization benefit is measurable
        _ = idx1.is_unique

        self.args = {
            "idx1": idx1,
            "idx2": idx2
        }

    def run(self):
        return self.args["idx1"].join(self.args["idx2"], how="outer").is_unique 

