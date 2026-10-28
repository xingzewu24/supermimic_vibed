"""Compatibility helpers for the pinned MuJoCo-Warp 3.11 collision implementation."""

from importlib.metadata import version


def configure_epa_horizon(capacity: int) -> None:
    """Allocate more EPA edges before CUDA graph capture, preserving GJK/EPA and its guards.

    3.11 allocates ``(naccdmax, MJ_MAX_EPAHORIZON)`` in collision_convex.
    _add_edge checks the array's actual shape, so no kernel or dependency file needs editing.
    This setting is process-wide and affects subsequently captured simulation graphs.
    """
    if capacity < 24:
        raise ValueError("epa_horizon must be at least 24")
    if version("mujoco-warp") != "3.11.0":
        raise RuntimeError("EPA capacity compatibility requires pinned mujoco-warp==3.11.0; "
                           "review the allocation API before using a different backend version")
    from mujoco_warp._src import collision_convex

    collision_convex.MJ_MAX_EPAHORIZON = capacity


def overflow_names(mask: int) -> str:
    from mujoco_warp._src.types import OverflowType

    names, known = [], 0
    for flag in OverflowType:
        known |= int(flag)
        if mask & int(flag):
            names.append(flag.name)
    if mask & ~known:
        names.append(hex(mask & ~known))
    return "|".join(names) or "NONE"
