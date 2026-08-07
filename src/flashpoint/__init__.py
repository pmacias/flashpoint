"""flashpoint: early-state wildfire severity classification.

Modules
-------
data_access   : discovering / reading converted WildfireSpreadTS HDF5 event data
db            : DuckDB schema creation and load helpers
labels        : deriving severity/triage classes from full event trajectories
features      : engineered early-state tabular features (day-1/day-2 cutoff)
"""

__version__ = "0.1.0"
