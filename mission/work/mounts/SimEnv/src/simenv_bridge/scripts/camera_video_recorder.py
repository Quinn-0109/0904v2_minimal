#!/usr/bin/env python3
"""Encode annotated robot-camera frames to an MP4 during the mission."""
import os

import cv2
import numpy as np
import rospy
from sensor_msgs.msg import Image
from std_msgs.msg import Bool


def image_msg_to_bgr(msg):
    if msg.encoding not in ("rgb8", "bgr8"):
        raise ValueError("unsupported image encoding: %s" % msg.encoding)
    expected = int(msg.width) * int(msg.height) * 3
    if len(msg.data) != expected:
        raise ValueError("invalid image buffer: expected %d bytes, got %d" % (expected, len(msg.data)))
    image = np.frombuffer(msg.data, dtype=np.uint8).reshape(int(msg.height), int(msg.width), 3)
    return image[:, :, ::-1].copy() if msg.encoding == "rgb8" else image.copy()


def should_write_frame(last_stamp, current_stamp, fps):
    return last_stamp is None or current_stamp - last_stamp >= (1.0 / float(fps)) - 1e-9


def open_video_writer(path, width, height, fps):
    directory = os.path.dirname(path) or "."
    os.makedirs(directory, exist_ok=True)
    writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), float(fps), (int(width), int(height)))
    if not writer.isOpened():
        raise RuntimeError("unable to open video writer: %s" % path)
    return writer


class CameraVideoRecorder:
    def __init__(self):
        self.output = rospy.get_param(
            "~out", "/workspace/SimEnv/results/scan_planner_demo/scanplanner_camera_fixed.mp4"
        )
        self.fps = float(rospy.get_param("~fps", 10.0))
        self.writer = None
        self.frame_size = None
        self.last_stamp = None
        self.closed = False
        rospy.Subscriber("/scanplanner/danger_debug_image", Image, self._image, queue_size=2, buff_size=2 ** 24)
        rospy.Subscriber("/scanplanner/three_floor_mission_complete", Bool, self._terminal, queue_size=1)
        rospy.Subscriber("/simenv/mission_abort", Bool, self._terminal, queue_size=1)
        rospy.on_shutdown(self.close)

    def _terminal(self, msg):
        if msg.data:
            self.close()

    def _image(self, msg):
        if self.closed:
            return
        stamp = msg.header.stamp.to_sec() if msg.header.stamp.to_sec() > 0.0 else rospy.get_time()
        if not should_write_frame(self.last_stamp, stamp, self.fps):
            return
        try:
            frame = image_msg_to_bgr(msg)
        except ValueError as error:
            rospy.logwarn_throttle(2.0, "camera recorder: %s", error)
            return
        if self.writer is None:
            self.frame_size = (frame.shape[1], frame.shape[0])
            self.writer = open_video_writer(self.output, self.frame_size[0], self.frame_size[1], self.fps)
            rospy.loginfo("camera recorder: %dx%d %.1f fps -> %s", self.frame_size[0], self.frame_size[1], self.fps, self.output)
        if (frame.shape[1], frame.shape[0]) != self.frame_size:
            frame = cv2.resize(frame, self.frame_size)
        self.writer.write(frame)
        self.last_stamp = stamp

    def close(self):
        if self.closed:
            return
        self.closed = True
        if self.writer is not None:
            self.writer.release()
            self.writer = None
            rospy.loginfo("camera recorder: wrote %s", self.output)


if __name__ == "__main__":
    rospy.init_node("camera_video_recorder", anonymous=False)
    CameraVideoRecorder()
    rospy.spin()
