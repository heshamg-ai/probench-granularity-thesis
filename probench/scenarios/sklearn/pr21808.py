from sklearn.linear_model import LogisticRegression

from probench.scenarios.scenario_base import PerfScenario

class PR21808Scenario(PerfScenario):
    """
    PR #21808 — Benchmark LogisticRegression.fit when using lbfgs solver for multiclass classification
    
    Parameters:
    - n: Number of samples
    - p: Number of features
    - c: Number of classes (benefits increase with more classes)
    """

    def setup(self):
        params = self.params
        n_samples = int(params["n"])
        n_features = int(params["p"])
        n_classes = int(params["c"])

        X = self.rng.standard_normal((n_samples, n_features))
        y = self.rng.integers(0, n_classes, size=n_samples)

        model = LogisticRegression(
            solver="lbfgs",
            max_iter=100,
            multi_class="multinomial",
        )

        self.args = {
            "model": model,
            "X": X,
            "y": y
        }

    def run(self):
        return self.args["model"].fit(self.args["X"], self.args["y"])
