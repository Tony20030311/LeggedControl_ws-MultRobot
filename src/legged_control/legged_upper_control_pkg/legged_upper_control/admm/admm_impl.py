"""ADMM implementation selector — one env var flips every gate to the C++ port.

    ADMM_IMPL=python (default) -> the Python golden reference
    ADMM_IMPL=cpp              -> admm_core_cpp (pybind11)

Gates import the swappable modules THROUGH this shim (C, ac, nq, ma, rti, es,
LaplacianFormation, reference/build_reference, AStarPlanner, and the fleet data
ARENAS/FORMATIONS/DEFAULT_GOALS + slot math).
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
    reference = _cpp.reference
    build_reference = _cpp.reference.build_reference
    AStarPlanner = _cpp.AStarPlanner
    ARENAS = _cpp.ARENAS
    FORMATIONS = _cpp.FORMATIONS
    DEFAULT_GOALS = _cpp.DEFAULT_GOALS
    rot2d = _cpp.rot2d
    centroid_slot_targets = _cpp.centroid_slot_targets
    min_cost_assignment = _cpp.min_cost_assignment
else:
    import constants                       # noqa: F401
    import admm_coordinator as ac          # noqa: F401
    import motion_adapter as ma            # noqa: F401
    import rti_linearizer as rti           # noqa: F401
    import node_subproblem as nq           # noqa: F401
    import edge_subproblem as es           # noqa: F401
    from core.formation import LaplacianFormation  # noqa: F401
    import reference                       # noqa: F401
    from reference import build_reference  # noqa: F401
    from core.planning import AStarPlanner  # noqa: F401

    sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(_HERE)), "scripts"))
    _FLEET_NAMES = ("ARENAS", "FORMATIONS", "DEFAULT_GOALS", "rot2d",
                    "centroid_slot_targets", "min_cost_assignment")

    def __getattr__(name):  # PEP 562
        # Lazy: the publisher itself imports this shim, so importing the
        # publisher eagerly here would be a circular import. Only the gates
        # access the fleet data through the shim.
        if name in _FLEET_NAMES:
            import ocs2_fleet_publisher as _pub
            for n in _FLEET_NAMES:
                globals()[n] = getattr(_pub, n)
            return globals()[name]
        raise AttributeError(name)
