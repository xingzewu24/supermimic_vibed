"""EPA workspaces and overflow diagnostics for the pinned collision backend."""

import pytest
import warp as wp
from mujoco_warp._src import collision_convex
from mujoco_warp._src.collision_gjk import Polytope, _add_edge

from supermimic.physics import configure_epa_horizon, overflow_names


@wp.kernel
def _epa_edges(horizon: wp.array2d[int], result: wp.array[int]):
    pt = Polytope()
    pt.horizon = horizon[0]
    pt.nhorizon = 0
    for edge in range(40):
        pt.nhorizon = _add_edge(pt, edge, edge + 1)
    result[0] = pt.nhorizon


@pytest.mark.parametrize("capacity, expected", [(24, -1), (128, 40)])
def test_epa_edge_workspace_supports_more_than_the_old_24_edge_limit(monkeypatch, capacity, expected):
    # Exercise the actual backend edge insertion/overflow guard, not a mock collider.
    monkeypatch.setattr(collision_convex, "MJ_MAX_EPAHORIZON", 24)
    configure_epa_horizon(capacity)
    horizon = wp.zeros((1, collision_convex.MJ_MAX_EPAHORIZON), dtype=int, device="cpu")
    result = wp.zeros(1, dtype=int, device="cpu")
    wp.launch(_epa_edges, dim=1, inputs=[horizon, result], device="cpu")
    assert result.numpy().tolist() == [expected]


def test_overflow_names_decode_epa_and_combined_or_unknown_bits():
    assert overflow_names(256) == "EPA_HORIZON"
    assert overflow_names(257) == "NEFC|EPA_HORIZON"
    assert overflow_names(0) == "NONE"
    assert overflow_names(1 << 20) == "0x100000"


def test_backend_version_change_requires_review(monkeypatch):
    monkeypatch.setattr("supermimic.physics.version", lambda package: "9.0.0")
    with pytest.raises(RuntimeError, match="review the allocation API"):
        configure_epa_horizon(128)


def test_invalid_capacity_is_rejected():
    with pytest.raises(ValueError, match="at least 24"):
        configure_epa_horizon(0)
