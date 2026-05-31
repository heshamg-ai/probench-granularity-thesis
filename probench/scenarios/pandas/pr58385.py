
import pandas as pd
import numpy as np
from probench.scenarios.scenario_base import PerfScenario


class PR58385Scenario(PerfScenario):
    """
    PR #58385 – Benchmark DataFrame.memory_usage() for a MultiIndex DataFrame

    Parameters:
    L : Level sizes for constructing the MultiIndex
    """

    def setup(self):
        params = self.params

        level_sizes = eval(params["L"])    
        n_levels = len(level_sizes)

        total_rows = int(np.prod(level_sizes))
        arrays = [
            np.repeat(np.arange(size), total_rows // size)
            for size in level_sizes
        ]

        names = [f"lvl_{i}" for i in range(n_levels)]
        mi = pd.MultiIndex.from_arrays(arrays, names=names)

        df = pd.DataFrame(
            {"value": self.rng.random(total_rows)},
            index=mi
        )

        self.args = {"df": df}

    def run(self):
        return self.args["df"].memory_usage()

