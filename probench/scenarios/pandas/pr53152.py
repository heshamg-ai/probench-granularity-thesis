import pandas as pd
import pyarrow as pa
from probench.scenarios.scenario_base import PerfScenario

class PR53152Scenario(PerfScenario):
    """
    PR #53152 – Benchmark Series.str.get for PyArrow-backed strings

    Parameters:
    - N: Series length
    """

    def setup(self):
        params = self.params
        N = int(params["N"])

        values = [f"str{i}" for i in range(N)]
        series = pd.Series(values, dtype=pd.ArrowDtype(pa.string()))

        self.args = {
            "series": series,
        }

    def run(self):
        return self.args["series"].str.get(1)
