# Occupied G4 recovery

Seed 824 (`fwd824b_s824_20260912_185652`) failed before driving to G4:
the recorded obstacle samples lie 0.266 and 0.344 m from the target, less
than the 0.38 m path clearance. Keeping that endpoint makes every detour
fail, regardless of which side the intermediate point is on.

After the existing local rescan fails, the sequencer now tries one local G4
replacement. It requires a completed G3, an oblique two-view policy and
fresh sensed points showing an obstacle within the endpoint clearance.
It tries 24 directions at offsets 0.15, 0.30, 0.45 and 0.60 m, checking:

- Direct approach against all available collision-band points, with the
  arrival tolerance added to the existing 0.38 m clearance.
- Existing geometry and minimum separation from actual G3 at the candidate
  and eight offsets on the arrival-tolerance circle.
- A final normal live path audit before adopting the candidate.

The original target and replacement are recorded in room evidence and an
`occupied_viewpoint_relocated` event. Runtime target records use the new
coordinates. Original planned polylines remain historical planning metadata.
Scan yaw, scan extent, detectors, ordinary transit and G3 selection are
unchanged. Failure to find a replacement retains the prior detour/failure
path. Actual arrival geometry checks still run.

A local test with the two recorded seed-824 samples and its geometry finds
(-3.8825, 15.0932), 0.45 m below the original G4. This does not replay the
full cloud or prove Gazebo success: the attached logs contain only the
nearest samples, and unsensed obstacles are unknown. Relocation may change
visibility despite preserving geometric contracts; detection quality and
mission duration need live verification. No world/truth file is read by
this recovery mechanism.
