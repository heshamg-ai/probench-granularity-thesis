import numpy as np
from sklearn.feature_selection import chi2

from probench.scenarios.scenario_base import PerfScenario

class PR21837Scenario(PerfScenario):
    """
    PR #21837 — Benchmark chi2 when y has many labels 

    Parameters:
    - n: Number of samples
    - p: Number of features
    - c: Number of unique classes
    """

    def setup(self):
        params = self.params
        n_samples = int(params["n"])
        n_features = int(params["p"])
        n_classes = int(params["c"])

        X = np.abs(self.rng.standard_normal((n_samples, n_features)))
        y = self.rng.integers(0, n_classes, size=n_samples)

        self.args = {"X": X, "y": y}

    def run(self):
        return chi2(self.args["X"], self.args["y"])
