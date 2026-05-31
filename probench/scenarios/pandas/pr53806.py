import numpy as np
import pandas as pd
from probench.scenarios.scenario_base import PerfScenario


class PR53806Scenario(PerfScenario):
    """
    PR #53806 - Benchmark `SeriesGroupBy.size` on a MultiIndex with already-sorted levels.

    Parameters:
    - R: Number of rows
    """

    def setup(self):
        params = self.params
        R = int(params["R"])

        level_size = int(np.sqrt(R))
        mi = pd.MultiIndex.from_product(
            [range(level_size), range(level_size)],
            names=["A", "B"]
        )
        mi = mi[:R]
        ser = pd.Series(self.rng.standard_normal(len(mi)), index=mi)

        self.args = {"ser": ser}

    def run(self):
        return self.args["ser"].groupby(["A", "B"]).size()
