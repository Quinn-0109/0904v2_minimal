#!/usr/bin/env python3
"""Stream the robot-mounted RGB camera to a reviewable video file."""

import json
import math
import os
import threading
import time

import cv2
import numpy as np
import rospy
from sensor_msgs.msg import Image
from std_msgs.msg import String


class RGBVideoRecorder:
    def __init__(self):
        self.output_dir = os.path.abspath(rospy.get_param("~output_dir"))
        self.video_dir = os.path.join(self.output_dir, "visualization")
        self.topic = rospy.get_param(
            "~rgb_topic", "/recording_camera/image_raw")
        self.fallback_topic = str(rospy.get_param(
            "~fallback_rgb_topic", "/real_sense/rgb/image_raw"))
        self.source_stall_timeout = max(0.5, float(rospy.get_param(
            "~source_stall_timeout_sec", 1.5)))
        self.target_fps = max(0.5, float(rospy.get_param(
            "~video_fps", 5.0)))
        self.preferred_codec = str(rospy.get_param(
            "~video_codec", "mp4v"))[:4]
        self.base_name = str(rospy.get_param(
            "~video_basename", "robot_camera_rgb"))
        self.overlay_phase = bool(rospy.get_param("~overlay_phase", True))
        self.lock = threading.RLock()
        self.writer = None
        self.video_path = None
        self.codec = None
        self.frame_size = None
        self.source_encoding = None
        self.received_frames = 0
        self.primary_received_frames = 0
        self.fallback_received_frames = 0
        self.written_frames = 0
        self.throttled_frames = 0
        self.decode_failures = 0
        self.resized_frames = 0
        self.first_source_stamp = None
        self.last_source_stamp = None
        self.last_written_stamp = None
        self.next_capture_stamp = None
        self.first_timeline_stamp = None
        self.last_timeline_stamp = None
        self.last_encoded_frame = None
        self.duplicated_frames = 0
        self.unfilled_timeline_gap_sec = 0.0
        self.maximum_gap_fill_frames = max(1, int(round(
            self.target_fps * float(rospy.get_param(
                "~maximum_gap_fill_sec", 30.0)))))
        self.second_floor_reached = False
        self.current_floor = 1
        self.phase_label = "STARTUP"
        self.stage_markers = []
        self.started_wall_time = time.time()
        self.finished = False
        self.error = None
        self.active_topic = None
        self.last_primary_timeline_stamp = None
        self.fallback_latched = False
        self.source_switches = []
        os.makedirs(self.video_dir, exist_ok=True)
        self.subscriber = rospy.Subscriber(
            self.topic, Image, self._on_primary_image, queue_size=2,
            buff_size=4 * 1024 * 1024)
        self.fallback_subscriber = None
        if self.fallback_topic and self.fallback_topic != self.topic:
            self.fallback_subscriber = rospy.Subscriber(
                self.fallback_topic, Image, self._on_fallback_image,
                queue_size=2, buff_size=4 * 1024 * 1024)
        self.baseline_state_subscriber = rospy.Subscriber(
            "/simenv/baseline_state", String, self._on_baseline_state,
            queue_size=8)
        self.stair_state_subscriber = rospy.Subscriber(
            "/simenv/stair_transition_state", String,
            self._on_stair_state, queue_size=8)
        self.second_floor_state_subscriber = rospy.Subscriber(
            "/simenv/second_floor_state", String,
            self._on_second_floor_state, queue_size=8)
        self.second_to_third_stair_state_subscriber = rospy.Subscriber(
            "/simenv/second_to_third_floor_stair_state", String,
            self._on_second_to_third_stair_state, queue_size=8)
        self.third_floor_state_subscriber = rospy.Subscriber(
            "/simenv/third_floor_state", String,
            self._on_third_floor_state, queue_size=8)
        rospy.on_shutdown(self.close)
        rospy.loginfo(
            "RGB video recording enabled: primary=%s fallback=%s "
            "target=%.1f FPS output=%s",
            self.topic, self.fallback_topic or "disabled", self.target_fps,
            os.path.join(self.video_dir, self.base_name + ".mp4"))

    @staticmethod
    def decode_image(message):
        """Decode common ROS 8-bit encodings into an OpenCV BGR frame."""
        encoding = str(message.encoding).lower()
        channels = {
            "rgb8": 3,
            "bgr8": 3,
            "rgba8": 4,
            "bgra8": 4,
            "mono8": 1,
            "8uc1": 1,
            "8uc3": 3,
            "8uc4": 4,
        }.get(encoding)
        if channels is None:
            raise ValueError("unsupported RGB image encoding: {}".format(
                message.encoding))
        height = int(message.height)
        width = int(message.width)
        step = int(message.step)
        if height <= 0 or width <= 0 or step < width * channels:
            raise ValueError("invalid image dimensions or row step")
        raw = np.frombuffer(message.data, dtype=np.uint8)
        required = height * step
        if raw.size < required:
            raise ValueError("image payload is shorter than height*step")
        rows = raw[:required].reshape(height, step)
        pixels = rows[:, :width * channels]
        if channels == 1:
            mono = pixels.reshape(height, width)
            return cv2.cvtColor(mono, cv2.COLOR_GRAY2BGR)
        frame = pixels.reshape(height, width, channels)
        if encoding == "rgb8":
            return cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        if encoding == "rgba8":
            return cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR)
        if encoding in ("bgra8", "8uc4"):
            return cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
        # Gazebo explicitly publishes rgb8. Treat generic 8UC3 as BGR, which
        # is the conventional OpenCV representation used by ROS cv_bridge.
        return np.ascontiguousarray(frame)

    @staticmethod
    def _message_stamp(message):
        try:
            stamp = float(message.header.stamp.to_sec())
            if stamp > 0.0:
                return stamp
        except (AttributeError, TypeError, ValueError):
            pass
        return time.monotonic()

    @staticmethod
    def due_timeline_slots(next_stamp, current_stamp, interval):
        """Return fixed-FPS slots due at the current ROS simulation time."""
        if next_stamp is None:
            return 1
        if current_stamp + 1e-4 < next_stamp:
            return 0
        return max(1, int(math.floor(
            (current_stamp - next_stamp + 1e-4) / interval)) + 1)

    def _timeline_stamp(self, message):
        try:
            stamp = float(rospy.Time.now().to_sec())
            if stamp > 0.0:
                return stamp
        except (AttributeError, TypeError, ValueError):
            pass
        return self._message_stamp(message)

    def _set_phase(self, label, source_state):
        with self.lock:
            if label == self.phase_label:
                return
            self.phase_label = label
            try:
                ros_stamp = float(rospy.Time.now().to_sec())
            except (AttributeError, TypeError, ValueError):
                ros_stamp = None
            video_time = (
                max(0.0, ros_stamp - self.first_timeline_stamp)
                if ros_stamp is not None and ros_stamp > 0.0 and
                self.first_timeline_stamp is not None else
                self.written_frames / self.target_fps)
            self.stage_markers.append({
                "label": label,
                "source_state": source_state,
                "ros_time_sec": ros_stamp,
                "video_time_sec": round(video_time, 3),
                "video_frame": int(round(video_time * self.target_fps)),
                "wall_time": time.time(),
            })

    def _on_baseline_state(self, message):
        state = str(message.data)
        prefix = "F{}".format(self.current_floor)
        self._set_phase("{}: {}".format(prefix, state), state)

    def _on_stair_state(self, message):
        state = str(message.data)
        if state.startswith("SECOND_FLOOR"):
            self.second_floor_reached = True
            self.current_floor = 2
            self._set_phase("F2: {}".format(state), state)
        elif state not in ("WAIT_F1", "FIRST_FLOOR_FINALIZING"):
            self._set_phase("STAIR: {}".format(state), state)

    def _on_second_floor_state(self, message):
        state = str(message.data)
        if not state or state == "WAIT_SECOND_FLOOR":
            return
        self.second_floor_reached = True
        self.current_floor = 2
        self._set_phase("F2: {}".format(state), state)

    def _on_second_to_third_stair_state(self, message):
        state = str(message.data)
        if not state or state == "WAIT_F1":
            return
        if state.startswith("THIRD_FLOOR"):
            self.current_floor = 3
            self._set_phase("F3: {}".format(state), state)
        else:
            self._set_phase("STAIR F2->F3: {}".format(state), state)

    def _on_third_floor_state(self, message):
        state = str(message.data)
        if not state or state == "WAIT_THIRD_FLOOR":
            return
        self.current_floor = 3
        self._set_phase("F3: {}".format(state), state)

    def _record_source_switch(self, previous, current, timeline_stamp,
                              reason):
        event = {
            "previous_topic": previous,
            "active_topic": current,
            "reason": reason,
            "ros_time_sec": timeline_stamp,
            "wall_time": time.time(),
        }
        self.source_switches.append(event)
        rospy.logwarn(
            "RGB recorder source switch: %s -> %s (%s)",
            previous or "none", current, reason)

    def _on_primary_image(self, message):
        timeline_stamp = self._timeline_stamp(message)
        with self.lock:
            self.primary_received_frames += 1
            self.last_primary_timeline_stamp = timeline_stamp
            if self.fallback_latched:
                return
            if self.active_topic != self.topic:
                previous = self.active_topic
                self.active_topic = self.topic
                self._record_source_switch(
                    previous, self.topic, timeline_stamp,
                    "dedicated_recording_camera_available")
        self._on_image(message, timeline_stamp=timeline_stamp)

    def _on_fallback_image(self, message):
        timeline_stamp = self._timeline_stamp(message)
        with self.lock:
            self.fallback_received_frames += 1
            primary_stalled = bool(
                self.last_primary_timeline_stamp is not None and
                timeline_stamp - self.last_primary_timeline_stamp >=
                self.source_stall_timeout)
            primary_never_started = bool(
                self.last_primary_timeline_stamp is None and
                time.time() - self.started_wall_time >=
                self.source_stall_timeout)
            if self.active_topic == self.topic and not primary_stalled:
                return
            if self.active_topic is None and not primary_never_started:
                return
            if self.active_topic != self.fallback_topic:
                previous = self.active_topic
                self.active_topic = self.fallback_topic
                # A fallback frame may simply win the startup race by a few
                # milliseconds.  Permit the dedicated camera to take over
                # later in that case.  Only a primary that was already live
                # and then stalled causes a permanent failover, preventing
                # source flapping during the remainder of the mission.
                self.fallback_latched = primary_stalled
                self._record_source_switch(
                    previous, self.fallback_topic, timeline_stamp,
                    ("primary_camera_stalled" if primary_stalled else
                     "primary_camera_startup_timeout"))
        self._on_image(message, timeline_stamp=timeline_stamp)

    def _annotate(self, frame):
        if not self.overlay_phase:
            return frame
        annotated = frame.copy()
        text = self.phase_label[:90]
        cv2.putText(
            annotated, text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX,
            0.58, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(
            annotated, text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX,
            0.58, (255, 255, 255), 1, cv2.LINE_AA)
        return annotated

    def _open_writer(self, width, height):
        candidates = []
        preferred_extension = (
            ".avi" if self.preferred_codec.upper() == "MJPG" else ".mp4")
        candidates.append((self.preferred_codec, preferred_extension))
        for candidate in (("mp4v", ".mp4"), ("MJPG", ".avi")):
            if candidate not in candidates:
                candidates.append(candidate)
        for codec, extension in candidates:
            path = os.path.join(self.video_dir, self.base_name + extension)
            writer = cv2.VideoWriter(
                path, cv2.VideoWriter_fourcc(*codec), self.target_fps,
                (width, height))
            if writer.isOpened():
                self.writer = writer
                self.video_path = path
                self.codec = codec
                self.frame_size = (width, height)
                return True
            writer.release()
        self.error = "no_supported_opencv_video_codec"
        rospy.logerr("Unable to open MP4 or AVI writer for robot RGB video.")
        return False

    def _on_image(self, message, timeline_stamp=None):
        source_stamp = self._message_stamp(message)
        timeline_stamp = (self._timeline_stamp(message)
                          if timeline_stamp is None else timeline_stamp)
        with self.lock:
            if self.finished:
                return
            self.received_frames += 1
            if (self.last_timeline_stamp is not None and
                    timeline_stamp < self.last_timeline_stamp - 1e-3):
                # Gazebo clock reset: begin a fresh sampling schedule while
                # keeping the same output stream open.
                self.next_capture_stamp = None
                self.first_timeline_stamp = timeline_stamp
            self.last_timeline_stamp = timeline_stamp
            interval = 1.0 / self.target_fps
            due_slots = self.due_timeline_slots(
                self.next_capture_stamp, timeline_stamp, interval)
            if due_slots <= 0:
                self.throttled_frames += 1
                return
            try:
                frame = self.decode_image(message)
            except (ValueError, cv2.error) as exc:
                self.decode_failures += 1
                self.error = str(exc)
                rospy.logwarn_throttle(5.0, "RGB video frame rejected: %s", exc)
                return
            height, width = frame.shape[:2]
            # Most video codecs require even dimensions. The simulated camera
            # is 640x480, but crop a possible odd final row/column safely.
            even_width = width - width % 2
            even_height = height - height % 2
            if even_width != width or even_height != height:
                frame = frame[:even_height, :even_width]
                width, height = even_width, even_height
            if self.writer is None and not self._open_writer(width, height):
                return
            if (width, height) != self.frame_size:
                frame = cv2.resize(
                    frame, self.frame_size, interpolation=cv2.INTER_AREA)
                self.resized_frames += 1
            frame = self._annotate(frame)
            if self.first_timeline_stamp is None:
                self.first_timeline_stamp = timeline_stamp
            if due_slots > self.maximum_gap_fill_frames:
                skipped = due_slots - self.maximum_gap_fill_frames
                self.unfilled_timeline_gap_sec += skipped * interval
                due_slots = self.maximum_gap_fill_frames
                self.next_capture_stamp = (
                    timeline_stamp - (due_slots - 1) * interval)
            if self.last_encoded_frame is not None:
                for _ in range(max(0, due_slots - 1)):
                    self.writer.write(self.last_encoded_frame)
                    self.duplicated_frames += 1
                    self.written_frames += 1
            self.writer.write(frame)
            self.written_frames += 1
            self.last_encoded_frame = frame
            self.source_encoding = str(message.encoding)
            self.first_source_stamp = (
                source_stamp if self.first_source_stamp is None
                else self.first_source_stamp)
            self.last_source_stamp = source_stamp
            self.last_written_stamp = source_stamp
            if self.next_capture_stamp is None:
                self.next_capture_stamp = timeline_stamp + interval
            else:
                self.next_capture_stamp += due_slots * interval

    def _write_metadata(self):
        ended_wall_time = time.time()
        source_duration = (
            max(0.0, self.last_source_stamp - self.first_source_stamp)
            if self.first_source_stamp is not None and
            self.last_source_stamp is not None else None)
        payload = {
            "schema": "simenv_robot_rgb_video_v1",
            "recording_complete": bool(
                self.video_path and self.written_frames > 0),
            "rgb_topic": self.topic,
            "fallback_rgb_topic": self.fallback_topic or None,
            "active_rgb_topic": self.active_topic,
            "source_stall_timeout_sec": self.source_stall_timeout,
            "primary_received_frames": self.primary_received_frames,
            "fallback_received_frames": self.fallback_received_frames,
            "source_switches": self.source_switches,
            "video_file": (
                os.path.relpath(self.video_path, self.output_dir)
                if self.video_path else None),
            "codec": self.codec,
            "target_fps": self.target_fps,
            "frame_size": list(self.frame_size) if self.frame_size else None,
            "source_encoding": self.source_encoding,
            "received_frames": self.received_frames,
            "written_frames": self.written_frames,
            "throttled_frames": self.throttled_frames,
            "decode_failures": self.decode_failures,
            "resized_frames": self.resized_frames,
            "duplicated_timeline_frames": self.duplicated_frames,
            "unfilled_timeline_gap_sec": self.unfilled_timeline_gap_sec,
            "source_duration_sec": source_duration,
            "ros_timeline_duration_sec": (
                max(0.0, self.last_timeline_stamp-
                    self.first_timeline_stamp)
                if self.first_timeline_stamp is not None and
                self.last_timeline_stamp is not None else None),
            "encoded_duration_sec": (
                self.written_frames / self.target_fps
                if self.written_frames else 0.0),
            "first_source_stamp_sec": self.first_source_stamp,
            "last_source_stamp_sec": self.last_source_stamp,
            "wall_started_at": self.started_wall_time,
            "wall_ended_at": ended_wall_time,
            "phase_overlay_enabled": self.overlay_phase,
            "stage_markers": self.stage_markers,
            "error": self.error,
        }
        destination = os.path.join(
            self.video_dir, self.base_name + "_video.json")
        temporary = destination + ".tmp"
        with open(temporary, "w") as stream:
            json.dump(payload, stream, indent=2, sort_keys=True)
        os.replace(temporary, destination)

    def close(self):
        with self.lock:
            if self.finished:
                return
            self.finished = True
            if self.writer is not None:
                self.writer.release()
                self.writer = None
            try:
                self._write_metadata()
            except OSError as exc:
                rospy.logerr("Failed to save RGB video metadata: %s", exc)
            if self.video_path:
                rospy.loginfo(
                    "RGB video saved: %s (%d frames, %.1f FPS)",
                    self.video_path, self.written_frames, self.target_fps)
            else:
                rospy.logwarn(
                    "RGB video recording ended without a writable frame; "
                    "check that %s was publishing.", self.topic)


def main():
    rospy.init_node("rgb_video_recorder")
    RGBVideoRecorder()
    rospy.spin()


if __name__ == "__main__":
    main()
