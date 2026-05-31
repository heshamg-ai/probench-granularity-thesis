import numpy as np
import pandas as pd
from probench.scenarios.scenario_base import PerfScenario

import io

class PR52548Scenario(PerfScenario):
    """
    PR #52548 – Benchmark pd.read_csv with pyarrow engine and temporal types.

    Parameters:
    - R: Number of rows
    - DC: Date columns
    - TT: Temporal type (timestamp/date)
    """

    def setup(self):
        params = self.params

        R = int(params["R"])
        DC = int(params["DC"])
        TT = params["TT"]

        data = {}
        
        for i in range(DC):
            if TT == "timestamp":
                data[f"date_col_{i}"] = pd.date_range('2020-01-01', periods=R, freq='h').strftime('%Y-%m-%d %H:%M:%S')
            elif TT == "date":
                data[f"date_col_{i}"] = pd.date_range('2020-01-01', periods=R, freq='D').strftime('%Y-%m-%d')
        
        for i in range(3):
            data[f"num_col_{i}"] = self.rng.random(R)

        df = pd.DataFrame(data)
        
        # Convert to CSV string
        csv_buffer = io.StringIO()
        df.to_csv(csv_buffer, index=False)
        csv_buffer.seek(0)
        csv_string = csv_buffer.getvalue()

        # Build parse_dates list
        parse_dates = [f"date_col_{i}" for i in range(DC)]

        self.args = {"csv" : csv_string, "parse_dates": parse_dates}

    def run(self):
        return pd.read_csv(
            io.StringIO(self.args["csv"]),
            engine="pyarrow",
            dtype_backend="pyarrow",
            parse_dates=self.args["parse_dates"]
        )
