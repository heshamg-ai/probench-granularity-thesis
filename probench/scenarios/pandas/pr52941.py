import numpy as np
import pandas as pd
from probench.scenarios.scenario_base import PerfScenario


class PR52941Scenario(PerfScenario):
    """
    PR #52941 – Benchmark DataFrame.filter with large items list.

    Parameters:
    - R: Number of rows
    - C: Number of columns
    - ILS: Items list size
    """

    def setup(self):
        params = self.params
        R = int(params["R"])
        C = int(params["C"])
        ILS = int(params["ILS"])

        data = {f"col_{i}": self.rng.random(R) for i in range(C)}
        df = pd.DataFrame(data)

        n_real = min(C, ILS)
        n_fake = max(0, ILS - n_real)

        items = [f"col_{i}" for i in range(n_real)]
        if n_fake:
            items.extend([f"fake_col_{i}" for i in range(n_fake)])

        self.args = {"df": df, "items": items}

    def run(self):
        return self.args["df"].filter(items=self.args["items"])

