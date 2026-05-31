import pandas as pd
import numpy as np
from probench.scenarios.scenario_base import PerfScenario


class PR59578Scenario(PerfScenario):
    """
    PR #59578 – Benchmark setting `MultiIndex.names` doesn't invalidate all cached operations.

    Parameters:
    - R: Number of rows
    - C: Number of columns
    - L: Number of levels in the MultiIndex
    """
    def needs_warmup(self) -> bool:
        return True

    def setup(self):
        params = self.params

        R = int(params["R"])
        L = int(params["L"])
        C = int(params["C"])

        levels = [np.arange(R) for _ in range(L)]
        names = [f"lvl{i}" for i in range(L)]

        mi = pd.MultiIndex.from_arrays(levels, names=names)

        df = pd.DataFrame(
            self.rng.random((R, C)),
            columns=[f"col_{i}" for i in range(C)],
            index=mi,
        )

        new_names = [name.upper() for name in names]

        self.args = {
            "df": df,
            "new_names": new_names,
        }

    def warmup(self):
        idx = self.args["df"].index

        _ = idx.is_monotonic_increasing
        _ = idx.is_unique
        _ = idx._engine
        _ = idx._lexsort_depth
        _ = idx.values
        _ = idx.get_level_values(0)

    def run(self):
        df = self.args["df"]

        # mutate
        df.index.names = self.args["new_names"]

        # trigger optimized path (cache reuse)
        _ = df.index.is_monotonic_increasing
        _ = df.index._engine
        _ = df.index.get_level_values(0)