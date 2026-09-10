#!/usr/bin/env python3
"""Ground-truth odometry relay: Gazebo truth -> FAST-LIO-compatible /Odometry.

Bypasses FAST-LIO entirely: the navigation chain (voxel mapper -> stabilized
odometry -> goal executor / floor managers / stair guides) is fed directly
from /gazebo/model_states so multi-round full-flow runs can measure the
remaining pipeline (exploration, stair policy, descent) without LIO drift,
degeneracy or frame-snap noise.

Frame contract:
  FAST-LIO publishes /Odometry in its camera_init frame.  For the truth
  bypass we publish the raw world-frame model pose (base link) with the
  conventional frame_ids so every consumer keeps working unchanged:
    frame_id = "camera_init", child_frame_id = "body"
  All managers are self-consistent in the odometry frame (goals are authored
  from the map built by the same stream), and the stair/corridor guides use
  Gazebo truth directly, so world-frame values agree with them exactly.

Only the model pose is relayed; twist comes from the same ModelStates sample
so the mapper's trajectory statistics and the planners' velocity logic see a
consistent stream.
"""

import math

import rospy
from gazebo_msgs.msg import ModelStates
from nav_msgs.msg import Odometry
from tf.transformations import quaternion_from_euler


def _quaternion_yaw(quat):
    x, y, z, w = quat.x, quat.y, quat.z, quat.w
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


class GroundTruthOdometry:
    def __init__(self):
        self._model = str(rospy.get_param("~model_name", "a1_gazebo"))
        self._odom_topic = str(rospy.get_param("~odom_topic", "/Odometry"))
        self._truth_topic = str(rospy.get_param(
            "~truth_topic", "/gazebo/model_states"))
        # Optional z lift: report base height (default) or add a sensor
        # offset to mimic FAST-LIO's IMU-frame height.
        self._z_offset = float(rospy.get_param("~z_offset_m", 0.0))
        # The model is spawned a few decimetres above the floor and falls to
        # its settled trunk height during the first ~1.5 s of sim time.  The
        # voxel mapper latches its planar height reference on the FIRST
        # /Odometry sample, so publishing that spawn-drop pose (z ~= 0.60)
        # leaves the whole climb compensation offset by the drop (~-0.29):
        # the F2 arrival area is then mapped ~0.29 m above its true height
        # and, once the F2 corridor-entry re-anchor drops the projection
        # slice to the true level, the phantom arrival-floor voxels land
        # inside the slice and every F2 start footprint looks occupied
        # (start_footprint_blocked / NO_VALID_FRONTIER).  Hold publishing
        # until the model has settled so the reference latches the true
        # body height and the climb offset stays ~0.
        self._settle_sec = float(rospy.get_param("~settle_sec", 3.0))
        self._frame_id = str(rospy.get_param("~frame_id", "camera_init"))
        self._child_frame_id = str(rospy.get_param(
            "~child_frame_id", "body"))
        self._publisher = rospy.Publisher(
            self._odom_topic, Odometry, queue_size=100)
        rospy.Subscriber(self._truth_topic, ModelStates,
                         self._on_model_states, queue_size=4)
        self._last_published = None
        self._publish_count = 0
        rospy.loginfo("Ground-truth odometry ready: %s -> %s (%s) "
                      "(settle hold %.1f s)",
                      self._truth_topic, self._odom_topic, self._model,
                      self._settle_sec)

    def _on_model_states(self, message):
        try:
            index = message.name.index(self._model)
        except ValueError:
            return
        pose = message.pose[index]
        twist = message.twist[index] if index < len(message.twist) else None
        if not all(math.isfinite(v) for v in
                   (pose.position.x, pose.position.y, pose.position.z,
                    pose.orientation.x, pose.orientation.y,
                    pose.orientation.z, pose.orientation.w)):
            return
        # Skip the spawn-drop transient (see _settle_sec above).  Sim time
        # does not advance while the world is paused, so this also correctly
        # waits out the launch-time pause.
        if rospy.Time.now().to_sec() < self._settle_sec:
            if self._publish_count == 0:
                rospy.loginfo("Ground-truth odometry holding until sim "
                              "%.1f s (model settling)", self._settle_sec)
            return
        odom = Odometry()
        # ModelStates has no header; stamp with the current sim time so the
        # registered-cloud relay can pair odom with each scan.
        odom.header.stamp = rospy.Time.now()
        odom.header.frame_id = self._frame_id
        odom.child_frame_id = self._child_frame_id
        odom.pose.pose.position.x = pose.position.x
        odom.pose.pose.position.y = pose.position.y
        odom.pose.pose.position.z = pose.position.z + self._z_offset
        odom.pose.pose.orientation = pose.orientation
        if twist is not None:
            odom.twist.twist = twist
        self._publisher.publish(odom)
        self._publish_count += 1
        if self._publish_count <= 5:
            yaw = _quaternion_yaw(pose.orientation)
            rospy.loginfo("Ground-truth odometry sample #%d at (%.3f, %.3f, "
                          "%.3f, yaw %.3f)", self._publish_count,
                          pose.position.x, pose.position.y,
                          pose.position.z, yaw)


if __name__ == "__main__":
    rospy.init_node("ground_truth_odometry")
    GroundTruthOdometry()
    rospy.spin()
