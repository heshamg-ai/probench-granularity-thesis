import pandas as pd
from probench.scenarios.scenario_base import PerfScenario



class PR53231Scenario(PerfScenario):
    """
    PR #53231 – Benchmark `DataFrame.merge` on datetime-like join keys.

    Parameters:
    - N: Number of rows in the left DataFrame. The right DataFrame uses `N//2` rows.
    """

    def setup(self):
        params = self.params
        N = int(params["N"])

        left_key = pd.date_range('2020-01-01', periods=N, freq='h')
        right_key = pd.date_range('2020-01-01', periods=N // 2, freq='h')
        N = int(params["N"])

        left = pd.DataFrame({
            'key': left_key,
            'value_left': self.rng.random(N)})

        right = pd.DataFrame({
            'key': right_key,
            'value_right': self.rng.random(N // 2)})

        self.args = {
            "left": left,
            "right": right
        }

    def run(self):
        return self.args["left"].merge(self.args["right"], on="key")
