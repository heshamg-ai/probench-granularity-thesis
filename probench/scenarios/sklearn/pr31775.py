from sklearn.metrics import median_absolute_error

from probench.scenarios.scenario_base import PerfScenario

class PR31775Scenario(PerfScenario):
    """
    PR #31775 — Benchmark _average_weighted_percentile optimization.

    Parameters:
    - n: Number of samples in the array
    - wt: Weight type 
    """

    def setup(self):
        params = self.params
        n_samples = int(params["n"])
        weight_type = params["wt"]

        y_true = self.rng.standard_normal(n_samples)
        y_pred = y_true + 0.1 * self.rng.standard_normal(n_samples)

        if weight_type == "integer":
            sample_weight = self.rng.integers(1, 100, size=n_samples).astype(float)
        else:
            sample_weight = self.rng.random(n_samples)

        self.args = {
            "y_true": y_true,
            "y_pred": y_pred,
            "sample_weight": sample_weight,
        }

    def run(self):
        return median_absolute_error(
            self.args["y_true"],
            self.args["y_pred"],
            sample_weight=self.args["sample_weight"],
        )
