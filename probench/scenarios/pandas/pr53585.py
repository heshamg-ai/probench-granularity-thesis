import pandas as pd
import pyarrow as pa
from probench.scenarios.scenario_base import PerfScenario


class PR53585Scenario(PerfScenario):
    """
    PR #53585 – Benchmark Series.str.split with expand=True for pyarrow strings.

    Parameters:
    - N: Series length
    """

    def setup(self):
        params = self.params
        N = int(params["N"])

        series = pd.Series(["foo|bar|baz"] * N, dtype=pd.ArrowDtype(pa.string()))

        self.args = {
            "series": series,
            "pat": "|"
        }

    def run(self):
        return self.args["series"].str.split(pat=self.args["pat"], expand=True)
