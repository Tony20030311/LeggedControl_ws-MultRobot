"""ADMM implementation selector — one env var flips every gate to the C++ port.

    ADMM_IMPL=python (default) -> the Python golden reference
    ADMM_IMPL=cpp              -> admm_core_cpp (pybind11)

Gates import the swappable modules THROUGH this shim (C, ac, nq, ma,
LaplacianFormation); data-prep modules (reference, AStarPlanner) stay Python.
"""

import os
import sys
import types

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.dirname(_HERE))

IMPL = os.environ.get("ADMM_IMPL", "python")

if IMPL == "cpp":
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
else:
    import constants                       # noqa: F401
    import admm_coordinator as ac          # noqa: F401
    import motion_adapter as ma            # noqa: F401
    import rti_linearizer as rti           # noqa: F401
    import node_subproblem as nq           # noqa: F401
    import edge_subproblem as es           # noqa: F401
    from core.formation import LaplacianFormation  # noqa: F401
