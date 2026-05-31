import numpy as np
import pandas as pd
from probench.scenarios.scenario_base import PerfScenario


class PR51498Scenario(PerfScenario):
    """
    PR #51498 – Benchmark DataFrame.round with numeric blocks.

    Parameters:
    - R: Number of rows
    - C: Number of columns
    - decimals: Number of decimal places
    """

    def setup(self):
        params = self.params
        
        R = int(params["R"])
        C = int(params["C"])
        decimals = int(params["decimals"])

        # Create DataFrame with mixed numeric and non-numeric columns
        data = {}
        
        # Add numeric columns
        for i in range(C):
            data[f"num_col_{i}"] = self.rng.random(R)

        df = pd.DataFrame(data)

        self.args = {"df": df, "decimals": decimals}

    def run(self):
        return self.args["df"].round(self.args["decimals"])
