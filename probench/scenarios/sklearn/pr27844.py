from sklearn.ensemble import HistGradientBoostingClassifier

from probench.scenarios.scenario_base import PerfScenario

class PR27844Scenario(PerfScenario):
    """
    PR #27844 — Benchmark HistGradientBoostingClassifier.predict when handling binary classification
    
    Parameters:
    - ntr: Number of training samples
    - npr: Number of samples to predict
    - p: Number of features
    """

    def setup(self):
        params = self.params
        n_samples_train = 10000
        n_samples_predict = int(params["npr"])
        n_features = 20

        X_train = self.rng.standard_normal((n_samples_train, n_features))
        y_train = self.rng.integers(0, 2, size=n_samples_train) #binary

        X_predict = self.rng.standard_normal((n_samples_predict, n_features))

        model = HistGradientBoostingClassifier(
            max_iter=50,
            max_depth=6,
            random_state=self.seed,
        )
        model.fit(X_train, y_train)

        self.args = {
            "model": model,
            "X_predict": X_predict
        }

    def run(self):
        return self.args["model"].predict(self.args["X_predict"])
