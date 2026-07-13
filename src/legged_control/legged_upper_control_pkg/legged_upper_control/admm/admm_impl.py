"""ADMM implementation binding — the C++ port (admm_core_cpp) is the sole path.

The Python golden reference and the ADMM_IMPL switch were retired at C6h; C++ is
now the only implementation. Gates still import the swappable symbols THROUGH this
shim (C, ac, nq, ma, rti, es, LaplacianFormation, reference/build_reference,
AStarPlanner, and the fleet data ARENAS/FORMATIONS/DEFAULT_GOALS + slot math), so
this stays the single binding surface.
"""

import os
import sys
import types

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.dirname(_HERE))

import admm_core_cpp as _cpp

constants = _cpp.constants
ac = _cpp.coordinator
ma = _cpp.motion_adapter
rti = _cpp.rti
LaplacianFormation = _cpp.LaplacianFormation
nq = types.SimpleNamespace(
    NodeSubproblem=_cpp.NodeSubproblem,
    h_obstacle=_cpp.h_obstacle,
    h_wall=_cpp.h_wall,
)
es = types.SimpleNamespace(EdgeSubproblem=_cpp.EdgeSubproblem)
reference = _cpp.reference
build_reference = _cpp.reference.build_reference
AStarPlanner = _cpp.AStarPlanner
ARENAS = _cpp.ARENAS
FORMATIONS = _cpp.FORMATIONS
DEFAULT_GOALS = _cpp.DEFAULT_GOALS
rot2d = _cpp.rot2d
centroid_slot_targets = _cpp.centroid_slot_targets
min_cost_assignment = _cpp.min_cost_assignment
