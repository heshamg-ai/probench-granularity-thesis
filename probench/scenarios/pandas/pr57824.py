import pandas as pd
from probench.scenarios.scenario_base import PerfScenario

class PR57824Scenario(PerfScenario):
    """
    PR #57824 – Benchmark RangeIndex.round

    Parameters:
    R : Specification of start, stop, and step used to build a RangeIndex
    D : Number of decimal places to round to
    """

    def setup(self):
        params = self.params

        start, stop, step = map(int, params["R"].split(","))
        D = int(params["D"])

        ri = pd.RangeIndex(start, stop, step)

        self.args = {
            "ri": ri,
            "decimals": D
        }

    def run(self):
        return self.args["ri"].round(self.args["decimals"])
