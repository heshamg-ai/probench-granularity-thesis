import pandas as pd
from probench.scenarios.scenario_base import PerfScenario
class PR57205Scenario(PerfScenario):
    """
    PR #57205 – Benchmark `DataFrame` construction from a dict with many columns.
    
    Parameters:
    - R: Number of rows
    - C: Number of columns
    """
    def setup(self):
        params = self.params

        R = int(params["R"])  
        C = int(params["C"])   
        
        base = self.rng.random(R)
        data = {i: base for i in range(C)}   

        columns = list(range(C))            

        self.args = {
            "data": data,
            "columns": columns
        }

    def run(self):
        return pd.DataFrame(self.args["data"], columns=self.args["columns"])

