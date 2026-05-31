import numpy as np
from sklearn.metrics import confusion_matrix

from probench.scenarios.scenario_base import PerfScenario

class PR9843Scenario(PerfScenario):
    """
    PR #9843 — Benchmark confusion_matrix when labels are integral

    Parameters:
    - n: Number of samples
    - c: Number of classes
    """

    def setup(self):
        params = self.params
        n_samples = int(params["n"])
        n_classes = int(params["c"])

        y_true = self.rng.integers(0, n_classes, size=n_samples)
        y_pred = self.rng.integers(0, n_classes, size=n_samples)

        labels = np.arange(n_classes)

        self.args = {
            "y_true": y_true,
            "y_pred": y_pred,
            "labels": labels
        }

    def run(self):
        return confusion_matrix(
            self.args["y_true"],
            self.args["y_pred"],
            labels=self.args["labels"]
        )
