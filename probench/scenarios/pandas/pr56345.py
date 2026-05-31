import pandas as pd
from probench.scenarios.scenario_base import PerfScenario


class PR56345Scenario(PerfScenario):
    """
    PR #56345 – Benchmark DataFrame.join on unordered CategoricalIndex.

    Parameters:
    - R : Number of rows in each DataFrame.
    - Cat : Number of categories in the CategoricalIndex.
    """

    def setup(self):
        params = self.params
        R = int(params["R"])
        Cat = int(params["Cat"])

        # Create data for indices
        categories = [f"cat_{i:03}" for i in range(Cat)]

        idx1 = pd.CategoricalIndex(
            self.rng.choice(categories, R, replace=True),
            categories=categories,
            ordered=False
        )

        idx2 = pd.CategoricalIndex(
            self.rng.choice(categories, R, replace=True),
            categories=reversed(categories), 
            ordered=False
        )

        df1 = pd.DataFrame({"val1": self.rng.random(R)}, index=idx1)
        df2 = pd.DataFrame({"val2": self.rng.random(R)}, index=idx2)

        self.args = {"df1": df1, "df2": df2}

    def run(self):
        return self.args["df1"].join(self.args["df2"])
