from scipy import sparse
from sklearn.linear_model import LinearRegression

from probench.scenarios.scenario_base import PerfScenario

class PR26207Scenario(PerfScenario):
    """
    PR #26207 — Benchmark LinearRegression.fit for sparse X or sparse y data

    Parameters:
    - n: Number of samples (large dataset)
    - p: Number of features
    """

    def setup(self):
        params = self.params
        n_samples = int(params["n"])
        n_features = int(params["p"])

        X_dense = self.rng.standard_normal((n_samples, n_features))
        X_dense[X_dense < 0.5] = 0
        X = sparse.csr_matrix(X_dense)
       
        y = self.rng.standard_normal(n_samples)
        model = LinearRegression()

        self.args = {
            "model": model,
            "X": X,
            "y": y
        }

    def run(self):
        return self.args["model"].fit(self.args["X"], self.args["y"])
