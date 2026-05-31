import warnings

from sklearn.decomposition import MiniBatchDictionaryLearning
from sklearn.exceptions import ConvergenceWarning

from probench.scenarios.scenario_base import PerfScenario

warnings.filterwarnings("ignore", category=ConvergenceWarning)

class PR25490Scenario(PerfScenario):
    """
    PR #25490 — Benchmark MiniBatchDictionaryLearning for small batch sizes

    Parameters:
    - n: Number of samples
    - p: Number of features
    - nc: Number of components
    """

    def setup(self):
        params = self.params
        n_samples = 1000
        n_features = int(params["p"])

        X = self.rng.standard_normal((n_samples, n_features))

        model = MiniBatchDictionaryLearning(
            n_components=int(params["nc"]),
            batch_size=1,  # PR optimization targets batch_size=1 (extreme case)
            max_iter=5,
            max_no_improvement=10000,
            tol=0,
            random_state=self.seed,
        )

        self.args = {
            "model": model,
            "X": X
        }

    def run(self):
        return self.args["model"].fit(self.args["X"])
