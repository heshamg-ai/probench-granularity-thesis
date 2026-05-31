# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2025, Siemens AG

from dataclasses import dataclass
from typing import List, Dict, Optional, Sequence, Mapping, Any
import itertools

@dataclass(frozen=True)
class ParamDef:
    """Definition of a single parameter axis (name + key + values)."""
    name: str
    key: str                    # formerly 'short'
    description: str            # formerly 'explanation'
    values: Sequence[str]       # keep strings; scenarios cast to param data type

    def validate(self):
        if not self.key:
            raise ValueError(f"ParamDef('{self.name}'): empty key")
        if not isinstance(self.values, (list, tuple)):
            raise TypeError(f"ParamDef('{self.name}'): values must be a Sequence[str]")

@dataclass(frozen=True)
class ParamView:
    """Concrete set of values for a given parameter combination."""
    values: Mapping[str, str]        # {"L": "1000", "DP": "0.1"}
    names: Mapping[str, str]         # {"L": "Series Length", ...}
    descriptions: Mapping[str, str]  # {"L": "Entries in the Series", ...}

    def __getitem__(self, key: str):
        return self.values[key]

    def get(self, key: str, default: Optional[str] = None):
        return self.values.get(key, default)

    def as_dict(self):
        return dict(self.values)

@dataclass(frozen=True)
class ParamGrid:
    """
    Unified parameter space for a performance scenario.

    ParamGrid holds a curated list of ParamDef axes for a given scenario
    and can enumerate the full Cartesian product of all parameter values.
    Each combination corresponds to one concrete benchmark configuration.
    """
    params: List[ParamDef] 

    def _expand(self, defs):
        if not defs:
            yield ParamView({}, {}, {})
            return

        for d in defs:
            d.validate()

        keys  = [d.key for d in defs]
        names = [d.name for d in defs]
        descs = [d.description for d in defs]
        grids = [list(d.values) for d in defs]

        for prod in itertools.product(*grids):
            values       = dict(zip(keys, prod))
            names_map    = dict(zip(keys, names))
            descs_map    = dict(zip(keys, descs))
            yield ParamView(values, names_map, descs_map)

    def pairs(self):
        """Yield all parameter combinations as casted dictionaries."""
        for p_view in self._expand(self.params):
            yield p_view.as_dict()

    @classmethod
    def from_json(cls, data: Dict[str, Any]):
        params = [ParamDef(**d) for d in data.get("params")]
        return cls(params=params)

