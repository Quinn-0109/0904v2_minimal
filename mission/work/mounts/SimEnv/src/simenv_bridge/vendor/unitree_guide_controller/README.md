# Unitree RL controller overlay

This directory vendors the controller files required by the mission from the
adjacent SimEnv reference workspace as of 2026-08-25.  They add policy-specific
`hybrid` device selection, safe policy reloads, fixed-stand handshakes, and
fresh locomotion-readiness signalling.  The Docker image's `CMakeLists.txt` is
retained because it contains the image-specific LibTorch/ROS prefix workaround.

`run_scanplanner_three_floor_rl.sh` copies only the explicitly enumerated files
into its owned container, rebuilds `unitree_guide`, and rejects the result if
the rebuilt `junior_ctrl` lacks hybrid-device support.
