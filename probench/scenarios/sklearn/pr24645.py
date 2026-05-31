from sklearn.neighbors import NearestCentroid

from probench.scenarios.scenario_base import PerfScenario

class PR24645Scenario(PerfScenario):
    """
    PR #24645 — Benchmark NearestCentroid.predict for large sample sizes and many classes

    Parameters:
    - npr: Number of samples to predict 
    - c: Number of classes
    """

    def setup(self):
        params = self.params
        n_samples_predict = int(params["npr"])
        n_classes = int(params["c"])

        X_train = self.rng.standard_normal((10000, 50))
        y_train = self.rng.integers(0, n_classes, size=10000)

        X_predict = self.rng.standard_normal((n_samples_predict, 50))

        model = NearestCentroid()
        model.fit(X_train, y_train)

        self.args = {
            "model": model,
            "X_predict": X_predict,
        }

    def run(self):
        return self.args["model"].predict(self.args["X_predict"])
