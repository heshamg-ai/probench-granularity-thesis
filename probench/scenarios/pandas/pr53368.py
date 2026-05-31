import pandas as pd
from probench.scenarios.scenario_base import PerfScenario



class PR53368Scenario(PerfScenario):
    """
    PR #53368 – Benchmark Index.get_indexer_for with Arrow temporal types.

    Parameters:
    - N: Index size
    """

    def setup(self):
        params = self.params
        N = int(params["N"])
 
        idx = pd.Index(range(N), dtype="timestamp[s][pyarrow]")
        target = idx[::2]

        self.args = {
            "idx": idx,
            "target": target
        }

    def run(self):
        return self.args["idx"].get_indexer_for(self.args["target"])