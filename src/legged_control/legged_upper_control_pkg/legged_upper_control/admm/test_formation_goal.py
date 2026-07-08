"""Unit tests for the /formation/goal virtual-centroid command math in
ocs2_fleet_publisher (centroid_slot_targets + min_cost_assignment). Pure functions,
no ROS. Run: python3 -c "import test_formation_goal as T,inspect; [getattr(T,n)() for n in dir(T) if n.startswith('test_')]"
"""
import os, sys
import numpy as np
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", "..", "scripts"))
from ocs2_fleet_publisher import centroid_slot_targets, min_cost_assignment, rot2d, FORMATIONS

V = FORMATIONS["V"]                       # [(0,0),(-0.7,0.5),(-0.7,-0.5)]


def test_centroid_is_the_commanded_point():
    """The 3 slots' centroid == the commanded goal (mean-centring), any yaw."""
    for yaw in (0.0, 0.7, np.pi / 2, -2.5, np.pi):
        slots = centroid_slot_targets((5.0, 1.0), V, yaw)
        c = np.mean(slots, axis=0)
        assert np.allclose(c, [5.0, 1.0], atol=1e-9), (yaw, c)


def test_shape_preserved_under_rotation():
    """Rotation is rigid: pairwise slot distances match the offsets' pairwise distances."""
    offs = [np.array(o) - np.mean(V, axis=0) for o in V]
    d_off = [np.linalg.norm(offs[a] - offs[b]) for a in range(3) for b in range(a + 1, 3)]
    slots = centroid_slot_targets((2.0, -3.0), V, 1.3)
    d_slot = [np.linalg.norm(slots[a] - slots[b]) for a in range(3) for b in range(a + 1, 3)]
    assert np.allclose(d_off, d_slot, atol=1e-9)


def test_yaw_zero_points_v_along_x():
    """yaw=0: the lead slot (offset (0,0)) is EAST of the two rear slots."""
    slots = centroid_slot_targets((0.0, 0.0), V, 0.0)
    assert slots[0][0] > slots[1][0] and slots[0][0] > slots[2][0]   # dog1 slot is front (+x)


def test_min_cost_assignment_picks_nearest():
    """Dogs sitting on the slots in SWAPPED order -> assignment returns the swap, not id."""
    slots = centroid_slot_targets((5.0, 0.0), V, 0.0)
    pos = [slots[2], slots[0], slots[1]]                 # dog order maps to slots 2,0,1
    assign = min_cost_assignment(pos, slots)
    assert assign == (2, 0, 1), assign
    # cost of chosen assignment is ~0 (dogs already on their nearest slots)
    cost = sum(float(np.dot(pos[k] - slots[assign[k]], pos[k] - slots[assign[k]])) for k in range(3))
    assert cost < 1e-12


def test_180_turn_reassigns_no_crossing():
    """Fleet faced east (yaw 0); now command a goal to the WEST (V flips 180). The nearest
    assignment must swap the rear slots so no dog crosses the whole formation."""
    east = centroid_slot_targets((6.0, 0.0), V, 0.0)     # where dogs currently sit
    west = centroid_slot_targets((0.0, 0.0), V, np.pi)   # flipped V slots
    assign = min_cost_assignment(east, west)
    # identity would send the front dog (east slot0) to the now-west slot0 = a full crossing;
    # the min-cost assignment must NOT be identity here.
    assert assign != (0, 1, 2), assign
