import pandas as pd
from probench.scenarios.scenario_base import PerfScenario


class PR56110Scenario(PerfScenario):
    """
    PR #56110 – Benchmark Series.str.get_dummies when dtype is string[pyarrow_numpy] or string[pyarrow]

    Parameters:
    - N: Series length
    - DT: String dtype (string[pyarrow_numpy]/string[pyarrow])
    """

    def setup(self):
        params = self.params
        
        N = int(params["N"])
        DT = params["DT"]

        values = [f"i-{i}" for i in range(N)]

        if DT == "string_pyarrow_numpy":
            series = pd.Series(values, dtype="string[pyarrow_numpy]")
        elif DT == "string_pyarrow":
            series = pd.Series(values, dtype="string[pyarrow]")

        series = series.str.join("|")
        self.args = {"series": series}

    def run(self):
        return self.args["series"].str.get_dummies("|")