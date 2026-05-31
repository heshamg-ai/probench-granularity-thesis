import pandas as pd
from probench.scenarios.scenario_base import PerfScenario


class PR53514Scenario(PerfScenario):
    """
    PR #53514 – Benchmark DataFrame.isin with large integer values.

    Parameters:
    - R: Number of rows
    - C: Number of columns
    """

    def setup(self):
        params = self.params
        R = int(params["R"])
        C = int(params["C"])

        data = self.rng.integers(0, 1000, size=(R, C))
        df = pd.DataFrame(data, dtype="int64[pyarrow]")

        # Small values list
        values = [1, 50]

        self.args = {
            "df": df,
            "values": values
        }

    def run(self):
        return self.args["df"].isin(self.args["values"])
