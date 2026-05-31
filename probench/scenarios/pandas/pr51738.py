import numpy as np
import pandas as pd
from probench.scenarios.scenario_base import PerfScenario


class PR51738Scenario(PerfScenario):
    """
    PR #51738 – Benchmark Index.is_unique access after slicing.

    Parameters:
    - N: Index Size
    """
    def needs_warmup(self) -> bool:
        return True
    
    def setup(self):
        params = self.params
        N = int(params["N"])

        idx = pd.Index(np.arange(N))
        self.args = {"idx": idx}

    def warmup(self):
        idx = self.args["idx"]

        # building cache
        _ = idx.is_monotonic_increasing
        _ = idx.is_unique

    def run(self):
        return self.args["idx"][:].is_unique