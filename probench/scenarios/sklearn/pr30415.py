import numpy as np
from sklearn.mixture import GaussianMixture

from probench.scenarios.scenario_base import PerfScenario

class PR30415Scenario(PerfScenario):
    """
    PR #30415 — Benchmark GaussianMixture.fit when input data is float32 dtype

    Parameters:
    - n: Number of samples
    """

    def setup(self):
        params = self.params
        n_samples = int(params["n"])

        X = self.rng.standard_normal((n_samples, 50)).astype(np.float32)

        model = GaussianMixture(
            n_components=10,
            covariance_type="full",
            max_iter=10,
            n_init=1,
            random_state=self.seed,
        )

        self.args = {
            "model": model,
            "X": X
        }

    def run(self):
        return self.args["model"].fit(self.args["X"])
