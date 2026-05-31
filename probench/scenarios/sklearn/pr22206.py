from sklearn.linear_model import QuantileRegressor

from probench.scenarios.scenario_base import PerfScenario

class PR22206Scenario(PerfScenario):
    """
    PR #22206 — Benchmark QuantileRegressor.fit with sparse matrix optimization.
    
    Parameters:
    - n: Number of samples (medium-sized dataset)
    - p: Number of features
    """

    def setup(self):
        params = self.params
        n_samples = int(params["n"])
        n_features = int(params["p"])

        X = self.rng.standard_normal((n_samples, n_features))
        true_coef = self.rng.standard_normal(n_features)
        y = X @ true_coef + 0.1 * self.rng.standard_normal(n_samples)

        model = QuantileRegressor(
            solver="highs",  # PR optimization targets highs solver
            quantile=0.75 #0.5
        )

        self.args = {
            "model": model,
            "X": X,
            "y": y
        }

    def run(self):
        return self.args["model"].fit(self.args["X"], self.args["y"])
