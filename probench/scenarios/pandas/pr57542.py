import pandas as pd
from probench.scenarios.scenario_base import PerfScenario


class PR57542Scenario(PerfScenario):
    """
    PR #57542 – Benchmark Series.str.extract with unnamed capture groups.

    Parameters:
    - G : Number of unnamed capture groups
    """

    N_REPS = 5000

    def setup(self):
        params = self.params
        G = int(params["G"])

        pattern = r"\s*".join(r"(\w+)" for _ in range(G))
        string = " ".join(f"word{i}" for i in range(G))
        series = pd.Series([string], dtype="object")

        self.args = {
            "series": series,
            "pattern": pattern,
        }

    def run(self):
        return [self.args["series"].str.extract(self.args["pattern"]) for _ in range(self.N_REPS)][-1]
