#!/usr/bin/env python3
"""ROS-free resolver for the junior_ctrl executable inside this workspace."""

import math
import os


class DevelSpaceResolutionError(SystemExit):
    pass


def _candidate_paths(devel_space):
    private = os.path.join(devel_space, ".private", "unitree_guide",
                           "lib", "unitree_guide", "junior_ctrl")
    plain = os.path.join(devel_space, "lib", "unitree_guide", "junior_ctrl")
    return (private, plain)


def _is_safe_executable(path, workspace):
    try:
        real = os.path.realpath(path)
        common = os.path.commonpath([real, workspace])
    except (OSError, ValueError):
        return False
    if common != workspace:
        return False
    return os.path.isfile(real) and os.access(real, os.X_OK)


def resolve_junior_ctrl(workspace, env=None):
    """Return the exact executable path or raise SystemExit.

    Preference order: SIMENV_DEVEL_SPACE (explicit), devel, devel_0803b (legacy fallback).
    Relative devel-space values are resolved below the workspace.  A candidate
    must be a regular executable whose realpath stays inside the workspace.
    """
    env = os.environ if env is None else env
    workspace = os.path.realpath(workspace)

    devel_spaces = []
    override = env.get("SIMENV_DEVEL_SPACE", "").strip()
    if override:
        devel_spaces.append(override if os.path.isabs(override)
                            else os.path.join(workspace, override))
    devel_spaces.append(os.path.join(workspace, "devel"))
    devel_spaces.append(os.path.join(workspace, "devel_0803b"))

    checked = []
    for devel_space in devel_spaces:
        for candidate in _candidate_paths(devel_space):
            if _is_safe_executable(candidate, workspace):
                return os.path.realpath(candidate)
            checked.append(candidate)
    raise DevelSpaceResolutionError(
        "junior_ctrl_not_found: no workspace-local executable; checked: {}"
        .format(", ".join(checked)))


def resolve_controller_settle_seconds(env=None):
    """Return the configured controller settle delay in seconds (default 0.0).

    Reads UNITREE_CONTROLLER_SETTLE_SECONDS; requires a finite float within
    [0.0, 5.0].  Invalid values raise SystemExit with a
    controller_settle_invalid message.
    """
    env = os.environ if env is None else env
    raw = env.get("UNITREE_CONTROLLER_SETTLE_SECONDS", "").strip()
    if not raw:
        return 0.0
    try:
        value = float(raw)
    except ValueError:
        raise SystemExit(
            "controller_settle_invalid: UNITREE_CONTROLLER_SETTLE_SECONDS="
            "{!r} is not a float".format(raw))
    if not math.isfinite(value):
        raise SystemExit(
            "controller_settle_invalid: UNITREE_CONTROLLER_SETTLE_SECONDS="
            "{!r} is not finite".format(raw))
    if value < 0.0 or value > 5.0:
        raise SystemExit(
            "controller_settle_invalid: UNITREE_CONTROLLER_SETTLE_SECONDS="
            "{!r} outside [0.0, 5.0]".format(raw))
    return value
