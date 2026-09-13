"""Seed 77 went down on an unaudited danger-rescan leg.

floor_0_room_1 plans g3 -> g4_path_1 (4.1, 14.15) -> g4 because
floor_0_room_1_coffee_table sits on the centreline (centreline_gap_m
-0.3).  The planned legs clear it by 0.66 m and 0.85 m.  The local danger
rescan drove g4 -> g3 and back STRAIGHT, skipping the bend: that line
passes 0.006 m from the very point the G3 audit had sensed on the table,
and neither rescan leg was run through the runtime 3D audit (11 audits in
the run, none after t=75.931).  It went through twice and ended upside
down, upright z -0.725, at a commanded 2.13 m/s.

Measured in fwd77_s77_20260912_234601.
"""
import math
import threading
from types import SimpleNamespace
import unittest

from test_three_floor_rl_mission import load_sequencer


G3_ACTUAL = (5.8092, 13.5322)
G4_PLANNED = (3.8825, 15.5432)
BEND = (4.1, 14.15)
SENSED_TABLE_POINT = (4.791, 14.603)
AUDIT_CLEARANCE_M = 0.38


def segment_distance(point, start, end):
    sx, sy = start
    ex, ey = end
    dx, dy = ex - sx, ey - sy
    length = dx * dx + dy * dy
    if length == 0.0:
        return math.hypot(point[0] - sx, point[1] - sy)
    t = max(0.0, min(1.0, ((point[0] - sx) * dx +
                           (point[1] - sy) * dy) / length))
    return math.hypot(point[0] - (sx + t * dx), point[1] - (sy + t * dy))


class RescanLegGeometryTest(unittest.TestCase):
    def test_the_planned_bend_clears_the_table_and_the_direct_line_does_not(self):
        self.assertGreater(
            segment_distance(SENSED_TABLE_POINT, G3_ACTUAL, BEND),
            AUDIT_CLEARANCE_M)
        self.assertGreater(
            segment_distance(SENSED_TABLE_POINT, BEND, G4_PLANNED),
            AUDIT_CLEARANCE_M)
        self.assertLess(
            segment_distance(SENSED_TABLE_POINT, G3_ACTUAL, G4_PLANNED),
            0.01)


class RescanLegAuditTest(unittest.TestCase):
    """Both rescan legs must be audited, and a failed audit must not drive."""

    def setUp(self):
        self.module = load_sequencer()
        self.source = self._read_source()

    def _read_source(self):
        from pathlib import Path
        path = (Path(__file__).resolve().parents[1] / "scripts" /
                "scanplanner_three_floor_goal_sequencer.py")
        return path.read_text("utf-8")

    def _rescan_block(self):
        start = self.source.index("local_danger_rescan_requested")
        end = self.source.index("local_danger_rescan_complete")
        return self.source[start:end]

    def _restore_block(self):
        start = self.source.index("local_danger_rescan_complete")
        return self.source[start:start + 3000]

    def test_the_outbound_leg_is_audited_before_it_drives(self):
        block = self._rescan_block()
        audit = block.index("self._runtime_3d_audit(")
        drive = block.index("_drive_direct_waypoint(")
        self.assertLess(audit, drive,
                        "the outbound rescan leg drives before it audits")

    def test_the_restore_leg_is_audited_before_it_drives(self):
        block = self._restore_block()
        audit = block.index("self._runtime_3d_audit(")
        drive = block.index("_drive_direct_waypoint(")
        self.assertLess(audit, drive,
                        "the restore leg drives before it audits")

    def test_the_check_is_the_bare_audit_and_never_the_driving_wrapper(self):
        # _ensure_runtime_3d_audit answers a failed audit by refreshing the
        # cloud and driving detour legs.  That is new motion for an
        # optional rescan and a new way for it to fail, so neither leg may
        # reach it; the bare check is the whole decision.
        for block in (self._rescan_block(), self._restore_block()):
            self.assertNotIn("_ensure_runtime_3d_audit", block)
            self.assertIn("self._runtime_3d_audit(", block)

    def test_a_failed_audit_keeps_the_room_instead_of_failing_the_route(self):
        for block in (self._rescan_block(), self._restore_block()):
            self.assertIn("return True, total", block)

    def _skip_body(self, block):
        """The audit-skip branch alone, up to the drive it guards."""
        start = block.index("if not self._runtime_3d_audit(")
        return block[start:block.index("_drive_direct_waypoint(", start)]

    def test_the_skip_does_not_touch_the_failure_flag(self):
        # The bare audit never sets self._failure, so the skip has nothing
        # to clear.  An earlier version of this cleared it unconditionally
        # and could have swallowed a real failure set elsewhere -- the
        # pre-existing direct_rl_no_progress path does clear it, on
        # purpose, and that one is outside this branch.
        for block in (self._rescan_block(), self._restore_block()):
            body = self._skip_body(block)
            self.assertNotIn("self._failure", body)
            self.assertIn("return True, total", body)

    def test_the_stale_reverse_segment_claim_is_gone(self):
        self.assertNotIn("exact reverse of the already", self.source)

    def test_both_skips_are_recorded(self):
        self.assertIn("local_danger_rescan_soft_skipped", self.source)
        self.assertIn("local_danger_rescan_restore_skipped", self.source)


if __name__ == "__main__":
    unittest.main()
