import numpy as np
from sklearn.ensemble._hist_gradient_boosting.binning import _find_binning_thresholds

from probench.scenarios.scenario_base import PerfScenario

class PR28102Scenario(PerfScenario):
    """
    PR #28102 — Benchmark _find_binning_thresholds large arrays with NaN values

    Parameters:
    - n: Number of samples
    - NAP: Fraction of NaN values in the data
    """

    MAX_BINS = 256  # Standard HGBT default

    def setup(self):
        params = self.params
        n_samples = int(params["N"])
        nan_fraction = float(params["NAP"])

        col_data = self.rng.standard_normal(n_samples).astype(np.float64)

        if nan_fraction > 0:
            nan_count = int(n_samples * nan_fraction)
            nan_indices = self.rng.choice(n_samples, size=nan_count, replace=False)
            col_data[nan_indices] = np.nan

        self.args = {
            "col_data": col_data,
            "max_bins": self.MAX_BINS,
        }

    def run(self):
        return _find_binning_thresholds(self.args["col_data"], max_bins=self.args["max_bins"])
