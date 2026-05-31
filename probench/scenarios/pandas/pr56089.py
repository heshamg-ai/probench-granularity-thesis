import pandas as pd
from probench.scenarios.scenario_base import PerfScenario

from pandas.api.types import CategoricalDtype

class PR56089Scenario(PerfScenario):
    """
    PR #56089 - Benchmark `pandas.get_dummies` on a Categorical Series.

    Parameters:
    N : Series length
    C : Number of unique categories
    """

    def setup(self):
        params = self.params
        
        N = int(params["N"])  
        C = int(params["C"]) 

        categories = [f"cat_{i}" for i in range(C)]
        values = self.rng.choice(categories, size=N)

        series = pd.Series(values, dtype=CategoricalDtype(categories))
        self.args = {"series": series}

    def run(self):
        return pd.get_dummies(self.args["series"], sparse=False)

