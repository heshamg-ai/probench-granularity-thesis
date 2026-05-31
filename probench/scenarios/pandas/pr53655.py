import pandas as pd
import pyarrow as pa
from probench.scenarios.scenario_base import PerfScenario


class PR53655Scenario(PerfScenario):
    """
    PR #53655 – Benchmark Series.str.get_dummies for pyarrow strings.

    Parameters:
    - K : Number of times to repeat the base pattern (final length = K * len(base_pattern)).
    """

    def setup(self):
        params = self.params
        K = int(params["K"])

        base_pattern = ["a|b|c", "a|b", "a|c", "b|c", "a", "b", "c"]
        pat_len = len(base_pattern)

        N = K * pat_len

        data = (base_pattern * (N // pat_len + 1))[:N]
        series = pd.Series(data, dtype=pd.ArrowDtype(pa.string()))

        self.args = {"series": series}

    def run(self):
        return self.args["series"].str.get_dummies()
