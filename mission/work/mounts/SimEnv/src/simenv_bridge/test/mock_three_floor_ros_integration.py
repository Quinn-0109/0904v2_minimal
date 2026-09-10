#!/usr/bin/env python3
"""ROS contract test for the three-floor SCAN goal sequencer.

This intentionally replaces physics and planners with immediate, deterministic
topic peers.  It verifies goal delivery, room-scan accounting, fresh policy
handshakes and the exact inter-floor tokens without claiming locomotion proof.
"""

import importlib.util
import json
import math
import os
from pathlib import Path
import shutil
import tempfile
import threading
import time
import unittest

from geometry_msgs.msg import PoseStamped, Twist
from nav_msgs.msg import Odometry
import rospy
import rostest
from sensor_msgs.msg import PointCloud2
from sensor_msgs import point_cloud2
from std_msgs.msg import Header
from std_msgs.msg import Bool, String


ROOT = Path(__file__).resolve().parents[1]
CONFIG = Path(os.environ.get(
    "MISSION_CONFIG", ROOT / "config" / "three_floor_rl_mission.json"))
SEQUENCER = ROOT / "scripts" / "scanplanner_three_floor_goal_sequencer.py"


def load_sequencer_module():
    spec = importlib.util.spec_from_file_location(
        "scanplanner_three_floor_goal_sequencer_integration", SEQUENCER
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class FakeThreeFloorPeers:
    def __init__(self):
        self._lock = threading.RLock()
        self._running = True
        self._pose = [0.0, -3.2, 0.34]
        self._yaw = math.pi / 2.0
        self.tokens = []
        self.policies = []

        self.odom_pub = rospy.Publisher(
            "/Odometry_gazebo", Odometry, queue_size=5, latch=True
        )
        self.cloud_pub = rospy.Publisher(
            "/registered_scan", PointCloud2, queue_size=1, latch=True
        )
        self.ready_pub = rospy.Publisher(
            "/locomotion_ready", Bool, queue_size=2, latch=True
        )
        self.camera_ready_pub = rospy.Publisher(
            "/simenv/recording_camera_ready", Bool, queue_size=1, latch=True
        )
        self.clock_ready_pub = rospy.Publisher(
            "/simenv/exploration_clock_ready", Bool, queue_size=1, latch=True
        )
        self.policy_status_pub = rospy.Publisher(
            "/rl_takeover_status", String, queue_size=2, latch=True
        )
        self.first_stair_pub = rospy.Publisher(
            "/simenv/stair_transition_state", String, queue_size=2, latch=True
        )
        self.second_stair_pub = rospy.Publisher(
            "/simenv/second_to_third_floor_stair_state",
            String,
            queue_size=2,
            latch=True,
        )

        rospy.Subscriber(
            "/move_base_simple/goal", PoseStamped, self._on_goal, queue_size=5
        )
        rospy.Subscriber(
            "/scanplanner/scan_cmd_vel", Twist, self._on_scan_command, queue_size=10
        )
        rospy.Subscriber(
            "/simenv/rl_policy_request", String, self._on_policy, queue_size=5
        )
        rospy.Subscriber(
            "/simenv/baseline_state", String, self._on_baseline, queue_size=5
        )
        rospy.Subscriber(
            "/simenv/second_floor_state", String, self._on_second, queue_size=5
        )
        rospy.Subscriber(
            "/simenv/third_floor_state", String, self._on_third, queue_size=5
        )

        self._publisher_thread = threading.Thread(
            target=self._publish_inputs, name="fake_three_floor_inputs", daemon=True
        )
        self._publisher_thread.start()

    def _publish_inputs(self):
        self.ready_pub.publish(Bool(data=True))
        self.camera_ready_pub.publish(Bool(data=True))
        self.clock_ready_pub.publish(Bool(data=True))
        while self._running and not rospy.is_shutdown():
            header = Header(stamp=rospy.Time.now(), frame_id="map")
            self.cloud_pub.publish(point_cloud2.create_cloud_xyz32(
                header, [(50.0, 50.0, 0.30)]))
            message = Odometry()
            message.header.stamp = rospy.Time.now()
            message.header.frame_id = "map"
            with self._lock:
                message.pose.pose.position.x = self._pose[0]
                message.pose.pose.position.y = self._pose[1]
                message.pose.pose.position.z = self._pose[2]
                message.pose.pose.orientation.z = math.sin(0.5 * self._yaw)
                message.pose.pose.orientation.w = math.cos(0.5 * self._yaw)
            self.odom_pub.publish(message)
            time.sleep(0.01)

    def _on_goal(self, message):
        q = message.pose.orientation
        target_yaw = 2.0 * math.atan2(float(q.z), float(q.w))
        with self._lock:
            self._pose = [
                float(message.pose.position.x),
                float(message.pose.position.y),
                float(message.pose.position.z) + 0.34,
            ]
            self._yaw = target_yaw

    def _on_scan_command(self, message):
        # Advance the contract-only robot faster than wall time, but retain a
        # real velocity integration step.  Treating m/s as metres per callback
        # can jump across a direct waypoint and make the turn-before-forward
        # controller oscillate forever around its target.
        integration_step_sec = 0.15
        rate = float(message.angular.z)
        with self._lock:
            if abs(rate) >= 1e-6:
                self._yaw += integration_step_sec * rate
            forward = float(message.linear.x)
            if abs(forward) >= 1e-6:
                # Contract physics is deliberately faster than wall time so
                # long optimized corridor legs fit inside rostest's deadline.
                distance = integration_step_sec * forward
                self._pose[0] += distance * math.cos(self._yaw)
                self._pose[1] += distance * math.sin(self._yaw)

    def _on_policy(self, message):
        policy = str(message.data)
        with self._lock:
            self.policies.append(policy)
        self.ready_pub.publish(Bool(data=False))
        self.policy_status_pub.publish(
            String(data="policy_reloaded:" + policy)
        )
        timer = threading.Timer(
            0.03, lambda: self.ready_pub.publish(Bool(data=True))
        )
        timer.daemon = True
        timer.start()

    def _remember(self, value):
        with self._lock:
            self.tokens.append(str(value))

    def _on_baseline(self, message):
        token = str(message.data).strip()
        self._remember(token)
        if token == "STAIR_WAIT_ZONE":
            self.first_stair_pub.publish(String(data="SECOND_FLOOR_REACHED"))

    def _on_second(self, message):
        token = str(message.data).strip()
        self._remember(token)
        if token == "SECOND_FLOOR_EXPLORATION_READY":
            self.first_stair_pub.publish(
                String(data="SECOND_FLOOR_HANDOFF_COMPLETE")
            )
        elif token == "SECOND_FLOOR_EXPLORATION_COMPLETE":
            self.second_stair_pub.publish(String(data="THIRD_FLOOR_REACHED"))

    def _on_third(self, message):
        token = str(message.data).strip()
        self._remember(token)
        if token == "THIRD_FLOOR_EXPLORATION_READY":
            self.second_stair_pub.publish(
                String(data="THIRD_FLOOR_HANDOFF_COMPLETE")
            )

    def stop(self):
        self._running = False
        self._publisher_thread.join(timeout=2.0)


class MockThreeFloorROSIntegrationTest(unittest.TestCase):
    def setUp(self):
        self.output_dir = tempfile.mkdtemp(prefix="scanplanner_mock_three_floor_")
        rospy.set_param("~mission_config", str(CONFIG))
        rospy.set_param("~output_dir", self.output_dir)
        rospy.set_param("~startup_timeout_sec", 5.0)
        rospy.set_param("~waypoint_timeout_sec", 15.0)
        # The fake odometry and command callbacks run on separate Python
        # threads; allow a complete accelerated in-place turn before declaring
        # that a direct waypoint has made no translational progress.
        rospy.set_param("~progress_timeout_sec", 2.0)
        rospy.set_param("~maximum_replans", 1)
        rospy.set_param("~policy_timeout_sec", 2.0)
        rospy.set_param("~handoff_timeout_sec", 3.0)
        # A long optimized scan can contain four 0.60 s stationary observation
        # pauses.  Keep this contract-only timeout above their real wall-time;
        # production retains its independent 30 s watchdog and 600 s mission
        # deadline.
        rospy.set_param("~room_scan_timeout_sec", 4.0)
        rospy.set_param("~rate", 50.0)
        self.peers = FakeThreeFloorPeers()

    def tearDown(self):
        self.peers.stop()
        shutil.rmtree(self.output_dir, ignore_errors=True)

    def test_full_topic_contract(self):
        module = load_sequencer_module()
        sequencer = module.ThreeFloorGoalSequencer()
        self.assertTrue(sequencer.run())

        paths = (
            Path(self.output_dir) / "tour_summary.json",
            Path(self.output_dir) / "second_floor" / "tour_summary.json",
            Path(self.output_dir) / "third_floor" / "tour_summary.json",
        )
        configured = json.loads(CONFIG.read_text(encoding="utf-8"))["floors"]
        expected_scans = 0
        configured_scan_total = 0
        for floor, path in zip(configured, paths):
            payload = json.loads(path.read_text(encoding="utf-8"))
            self.assertTrue(payload["success"], payload)
            scans = [
                item for item in payload["waypoints"]
                if item.get("scan_completed") is True
            ]
            configured_count = sum(
                (waypoint.get("scan") is True or
                 str(waypoint.get("note", "")).endswith("_center"))
                for waypoint in floor["waypoints"])
            self.assertEqual(len(scans), configured_count)
            expected_scans += len(scans)
            configured_scan_total += configured_count
        self.assertEqual(expected_scans, configured_scan_total)
        if any(waypoint.get("scan") is True
               for floor in configured for waypoint in floor["waypoints"]):
            self.assertEqual(expected_scans, 24)
        first_tour = json.loads(paths[0].read_text(encoding="utf-8"))
        entrance = [
            item for item in first_tour["waypoints"]
            if str(item.get("note", "")).startswith("main_entrance_")
        ]
        self.assertEqual(len(entrance), 3)
        self.assertTrue(all(
            item.get("controller") == "direct_forward_stair_rl"
            for item in entrance
        ))
        self.assertGreaterEqual(len(self.peers.policies), 2)
        self.assertIn("stair", Path(self.peers.policies[0]).name)
        self.assertIn("plane", Path(self.peers.policies[1]).name)

        route = json.loads(
            (Path(self.output_dir) / "scanplanner_route_summary.json").read_text(
                encoding="utf-8"
            )
        )
        self.assertEqual(route["status"], "completed")
        self.assertEqual(route["floors_completed"], [1, 2, 3])
        for token in (
            "STAIR_WAIT_ZONE",
            "SECOND_FLOOR_EXPLORATION_READY",
            "SECOND_FLOOR_EXPLORATION_COMPLETE",
            "THIRD_FLOOR_EXPLORATION_READY",
            "THIRD_FLOOR_EXPLORATION_COMPLETE",
        ):
            self.assertIn(token, self.peers.tokens)


if __name__ == "__main__":
    rospy.init_node("mock_three_floor_ros_integration", anonymous=False)
    rostest.rosrun(
        "simenv_bridge",
        "mock_three_floor_ros_integration",
        MockThreeFloorROSIntegrationTest,
    )
