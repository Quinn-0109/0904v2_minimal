#!/usr/bin/env python3
"""ROS-free resolver for the junior_ctrl executable used by this bundle.

0904v2_minimal keeps exactly one validated controller build inside the bundle
(SimEnv-master/.portable/devel/lib/unitree_guide/junior_ctrl) plus the gazebo
plugins in the same devel space.  The exploration mount has no unitree_guide
package of its own, so the locomotion wrapper resolves the prebuilt binary
here instead of guessing a workspace layout.

Preference order (first existing executable wins):
  1. $SIMENV_EXPL_DEVEL_SPACE (set by run_exploration.sh; may be absolute or
     relative to the exploration mount root)
  2. <mount root>/devel, <mount root>/devel_0803b (legacy layouts kept for
     parity with the standalone delivery)
Each candidate is tried as both the catkin_tools private layout
(.private/unitree_guide/lib/...) and the plain catkin_make layout
(lib/unitree_guide/...).
"""

import os


class DevelSpaceResolutionError(SystemExit):
    pass


def _candidate_paths(devel_space):
    private = os.path.join(devel_space, ".private", "unitree_guide",
                           "lib", "unitree_guide", "junior_ctrl")
    plain = os.path.join(devel_space, "lib", "unitree_guide", "junior_ctrl")
    return (private, plain)


def _is_executable(path):
    return os.path.isfile(path) and os.access(path, os.X_OK)


def resolve_junior_ctrl(mount_root, env=None):
    """Return the exact junior_ctrl executable path or raise SystemExit."""
    env = os.environ if env is None else env
    mount_root = os.path.realpath(mount_root)

    devel_spaces = []
    override = env.get("SIMENV_EXPL_DEVEL_SPACE", "").strip()
    if override:
        devel_spaces.append(override if os.path.isabs(override)
                            else os.path.join(mount_root, override))
    devel_spaces.append(os.path.join(mount_root, "devel"))
    devel_spaces.append(os.path.join(mount_root, "devel_0803b"))

    checked = []
    for devel_space in devel_spaces:
        for candidate in _candidate_paths(devel_space):
            if _is_executable(candidate):
                return os.path.realpath(candidate)
            checked.append(candidate)
    raise DevelSpaceResolutionError(
        "junior_ctrl_not_found: set SIMENV_EXPL_DEVEL_SPACE to a devel space "
        "containing lib/unitree_guide/junior_ctrl; checked: {}"
        .format(", ".join(checked)))
