#!/usr/bin/env python3
"""Cluster independent RGB-D red observations in the online map frame."""
import json
import math
import os
import time

import rospy
from std_msgs.msg import String


class HazardTracker:
    def __init__(self):
        # FAST-LIO's local map can shift a few decimetres while the robot
        # turns in place.  Keep repeated views of one physical ball together
        # instead of publishing a second confirmed copy as a false positive.
        self.radius = float(rospy.get_param("~cluster_radius_m", 1.05))
        # During an in-place camera sweep FAST-LIO can move the projected
        # position of one physical ball by more than the ordinary map-space
        # cluster radius. Keep a narrowly bounded continuity association for
        # consecutive, strict, high-confidence observations in the *same*
        # room. This does not relax the detector or confirmation gates; it
        # only prevents one ball from becoming two one-hit clusters.
        self.continuity_radius = float(rospy.get_param(
            "~continuity_cluster_radius_m", 1.75))
        self.continuity_seconds = float(rospy.get_param(
            "~continuity_cluster_seconds", 2.0))
        # A confirmed sphere can be seen again while the robot is leaving one
        # room or entering its neighbour. Room labels are route context, not
        # physical identity, so suppress only close 3-D duplicates across a
        # room boundary. The vertical gate prevents merging floors.
        self.cross_room_duplicate_radius = float(rospy.get_param(
            "~cross_room_duplicate_radius_m", 0.60))
        self.cross_room_duplicate_max_vertical = float(rospy.get_param(
            "~cross_room_duplicate_max_vertical_m", 0.80))
        self.minimum = int(rospy.get_param("~minimum_observations", 3))
        self.minimum_baseline = float(rospy.get_param(
            "~minimum_view_baseline_m", .12))
        self.minimum_yaw_diversity = math.radians(float(rospy.get_param(
            "~minimum_view_yaw_diversity_deg", 8.0)))
        # A fast sweep can expose a distant 0.30 m ball for only two adjacent
        # RGB frames. Keep this exception narrower than the ordinary three-hit
        # gate: strict sphere shape, high confidence, metric-size consistency,
        # a short time span, and independently separated headings/positions.
        self.strict_pair_minimum_confidence = float(rospy.get_param(
            "~strict_pair_minimum_confidence", 0.90))
        self.strict_pair_minimum_yaw_diversity = math.radians(float(
            rospy.get_param("~strict_pair_minimum_yaw_diversity_deg", 3.0)))
        self.strict_pair_minimum_diameter = float(rospy.get_param(
            "~strict_pair_minimum_diameter_m", 0.24))
        self.strict_pair_maximum_diameter = float(rospy.get_param(
            "~strict_pair_maximum_diameter_m", 0.34))
        self.strict_pair_maximum_span = float(rospy.get_param(
            "~strict_pair_maximum_span_sec", 0.75))
        self.mixed_pair_maximum_span = float(rospy.get_param(
            "~mixed_strict_partial_maximum_span_sec", 0.50))
        self.mixed_pair_minimum_yaw_diversity = math.radians(float(
            rospy.get_param(
                "~mixed_strict_partial_minimum_yaw_diversity_deg", 4.0)))
        # Translational verification is deliberately optional.  A room's
        # in-place visual sweep already supplies five viewing directions; on
        # the 180 s profile, moving again toward every first red blob costs
        # more mission time than it adds confirmation evidence.
        self.enable_verify_goal = bool(rospy.get_param(
            "~enable_verify_goal", False))
        self.output_dir = rospy.get_param("~output_dir", os.getcwd())
        self.log_path = os.path.join(self.output_dir, "logs", "hazard_candidates.jsonl")
        self.result_path = rospy.get_param("~result_file", os.path.join(self.output_dir, "detected_danger.json"))
        self.confirmed_path = rospy.get_param("~confirmed_file", os.path.join(self.output_dir, "confirmed_hazards.json"))
        self.diagnostic_path = rospy.get_param("~diagnostic_file", os.path.join(self.output_dir, "danger_detection_diagnostics.json"))
        # Persist the coordinate contract used by the RGB-D ray transform so
        # offline scoring does not try to apply the F1 FAST-LIO->truth
        # correction a second time.  This is metadata only; classification
        # and online confirmation remain image/depth based.
        self.position_frame = str(rospy.get_param(
            "~position_frame", "fastlio_map")).strip() or "fastlio_map"
        os.makedirs(os.path.dirname(self.log_path), exist_ok=True)
        self.clusters, self.next_id = [], 1
        self.room_search_stats = {}
        self.last_stats_persist = -math.inf
        self.pub = rospy.Publisher("/simenv/confirmed_hazards", String, queue_size=10, latch=True)
        # This compact online diagnostic lets the route manager distinguish a
        # genuinely empty visual fan from a strict sphere that merely needs
        # one more viewing angle for confirmation.  It contains no truth or
        # layout information and never commands motion by itself.
        self.status_pub = rospy.Publisher("/simenv/hazard_candidate_status",
                                          String, queue_size=10)
        self.verify_pub = rospy.Publisher("/simenv/visual_verify_request", String, queue_size=10)
        rospy.Subscriber("/simenv/red_ball_observations", String, self.on_observation, queue_size=50)
        rospy.Subscriber("/simenv/red_ball_search_status", String,
                         self.on_search_status, queue_size=20)

    def on_search_status(self, message):
        try:
            payload = json.loads(message.data)
            room_id = str(payload["room_id"])
        except (TypeError, ValueError, KeyError):
            return
        if room_id:
            # The detector publishes cumulative counters per room.  Retain
            # only the newest snapshot so diagnostics do not double count.
            self.room_search_stats[room_id] = payload
            now = time.monotonic()
            if now - self.last_stats_persist >= 2.0:
                self.last_stats_persist = now
                self.persist()

    def persist(self):
        confirmed = [self.public(c) for c in self.clusters if c["confirmed"]]
        value = {"generated_at": time.time(),
                 "position_frame": self.position_frame,
                 "confirmed_hazards": confirmed,
                 # Preserve the established competition/result-file schema.
                 "detected_danger_sources": [
                     {"position": [c["position"]["x"], c["position"]["y"], c["position"]["z"]],
                      "confidence": c["confidence"],
                      "observation_stamp": c.get("observation_stamp"),
                      "room_id": c.get("room_id"),
                      "source": c.get("source", "rgbd_active_search")}
                     for c in confirmed]}
        try:
            with open(self.result_path, "w", encoding="utf-8") as f:
                json.dump(value, f, indent=2, sort_keys=True)
            if self.confirmed_path != self.result_path:
                with open(self.confirmed_path, "w", encoding="utf-8") as f:
                    json.dump(value, f, indent=2, sort_keys=True)
            room_stats = list(self.room_search_stats.values())
            diagnostics = {"schema": "simenv_active_rgbd_tracker_v1",
                           "diagnostics": {"frames_outside_room_skipped": 0,
                                           "frames_in_room_processed": sum(
                                               int(item.get("frames", 0))
                                               for item in room_stats),
                                           "frames_with_red_pixels": sum(
                                               int(item.get("red_pixel_frames", 0))
                                               for item in room_stats),
                                           "red_contours_seen": sum(
                                               int(item.get("red_contours_seen", 0))
                                               for item in room_stats),
                                           "candidate_events": [
                                               {"stamp": c.get("confirmed_stamp"),
                                                "position": [c["x"], c["y"], c["z"]],
                                                "status": "confirmed", "room_id": c.get("room_id")}
                                               for c in self.clusters if c["confirmed"]]}}
            with open(self.diagnostic_path, "w", encoding="utf-8") as f:
                json.dump(diagnostics, f, indent=2, sort_keys=True)
        except OSError as exc:
            rospy.logwarn_throttle(10.0, "hazard result unavailable: %s", exc)
        self.pub.publish(String(data=json.dumps(value, sort_keys=True)))

    def publish_room_status(self, room_id):
        room_id = str(room_id or "")
        if not room_id:
            return
        pending = []
        for cluster in self.clusters:
            if str(cluster.get("room_id") or "") != room_id or cluster["confirmed"]:
                continue
            pending.append({"id": cluster["id"], "strict_hits": int(
                cluster.get("strict_hits", 0)), "profile_relaxed_hits": int(
                cluster.get("profile_relaxed_hits", 0)), "hits": int(
                cluster["hits"]), "confidence": round(float(cluster["confidence"]), 3)})
        self.status_pub.publish(String(data=json.dumps({
            "room_id": room_id, "timestamp": round(time.time(), 6),
            "pending_candidates": pending}, sort_keys=True)))

    @staticmethod
    def public(c):
        # A FAST-LIO map may drift slightly during a five-view sweep.  A mean
        # in that moving frame cannot be paired with any one camera pose for
        # offline truth evaluation.  Publish the strongest centred RGB-D
        # observation together with its own stamp; clustering still uses the
        # running mean above, so this does not weaken online association.
        representative = c.get("representative") or {
            "x": c["x"], "y": c["y"], "z": c["z"], "timestamp": None,
            "source": "rgbd_active_search"}
        value = {"id": c["id"], "position": {"x": round(representative["x"], 4), "y": round(representative["y"], 4), "z": round(representative["z"], 4)},
                "observations": c["hits"], "confidence": round(c["confidence"], 3), "confirmed": c["confirmed"], "room_id": c.get("room_id"),
                "source": representative.get("source", "rgbd_active_search")}
        if representative.get("timestamp") is not None:
            value["observation_stamp"] = round(float(representative["timestamp"]), 6)
        return value

    def view_evidence(self, cluster):
        views = cluster.get("views", [])
        baseline, yaw_diversity = 0.0, 0.0
        for index, first in enumerate(views):
            for second in views[index + 1:]:
                baseline = max(baseline, math.hypot(first[0] - second[0],
                                                    first[1] - second[1]))
                delta = abs((first[2] - second[2] + math.pi) %
                            (2.0 * math.pi) - math.pi)
                yaw_diversity = max(yaw_diversity, delta)
        return baseline, yaw_diversity

    def strict_pair_observation_eligible(self, obs, pixel):
        try:
            diameter = float(obs.get("diameter_m", math.nan))
            confidence = float(obs.get("confidence", 0.0))
        except (TypeError, ValueError):
            return False
        return bool(
            str(pixel.get("shape_gate", "strict")) == "strict" and
            not bool(obs.get("profile_relaxed", False)) and
            confidence >= self.strict_pair_minimum_confidence and
            math.isfinite(diameter) and
            self.strict_pair_minimum_diameter <= diameter <=
            self.strict_pair_maximum_diameter)

    def is_confirmed(self, cluster):
        baseline, yaw_diversity = self.view_evidence(cluster)
        strict_hits = cluster.get("strict_hits", 0)
        # A close, clean sphere can be visible for only two RGB frames while
        # the camera sweeps past the room boundary.  Run
        # three_floor_check_run_f2_f3_fix_1 saw D8 twice with 0.929 confidence
        # and a 0.221 m viewpoint baseline, but the legacy three-hit gate
        # discarded it.  Retain the multi-view requirement and admit only a
        # high-confidence strict-shape pair; relaxed/oblique red contours and
        # single-frame red-box distractors remain ineligible.
        strict_pair_views = cluster.get("strict_pair_views", [])
        pair_baseline, pair_yaw_diversity = self.view_evidence({
            "views": strict_pair_views})
        pair_stamps = cluster.get("strict_pair_stamps", [])
        pair_span = ((max(pair_stamps) - min(pair_stamps))
                     if len(pair_stamps) >= 2 else math.inf)
        if (cluster.get("strict_pair_hits", 0) >= 2 and
                pair_span <= self.strict_pair_maximum_span and
                (pair_baseline >= self.minimum_baseline or
                 pair_yaw_diversity >=
                 self.strict_pair_minimum_yaw_diversity)):
            return True
        # A furniture edge can expose only one clean sphere silhouette.
        # Corroborate it with one same-cluster partial-sphere frame, but never
        # promote a partial-only red cuboid sequence.
        observation_stamps = cluster.get("observation_stamps", [])
        mixed_span = ((max(observation_stamps) - min(observation_stamps))
                      if len(observation_stamps) >= 2 else math.inf)
        if (cluster.get("strict_pair_hits", 0) >= 1 and
                cluster.get("strict_hits", 0) >= 1 and
                cluster.get("profile_relaxed_hits", 0) >= 1 and
                cluster.get("hits", 0) >= 2 and
                mixed_span <= self.mixed_pair_maximum_span and
                yaw_diversity >= self.mixed_pair_minimum_yaw_diversity):
            return True
        # A strict close-range observation must not inherit the more demanding
        # remote gate merely because an earlier frame was profile-relaxed.
        if strict_hits >= self.minimum:
            return (baseline >= self.minimum_baseline or
                    yaw_diversity >= self.minimum_yaw_diversity)
        if cluster.get("profile_relaxed_hits", 0) > 0:
            # A permissive oblique/partial silhouette is corroborating
            # evidence only.  In run112 the distant red cuboid X89 supplied
            # eleven mutually separated oblique frames while sparse RGB-D
            # prevented a signed convexity measurement; the former rule
            # therefore confirmed a planar distractor with zero strict
            # sphere observations.  Require one ordinary sphere-shaped frame
            # before the bounded five-view relaxed route can confirm.  All
            # true hazards in strict runs 106--108 had >=3 strict frames, so
            # this closes the observed FP without using truth coordinates.
            return (strict_hits >= 1 and
                    cluster["profile_relaxed_hits"] >= max(self.minimum, 5) and
                    yaw_diversity >= max(self.minimum_yaw_diversity,
                                         math.radians(15.0)))
        return (cluster["hits"] >= self.minimum and
                (baseline >= self.minimum_baseline or
                 yaw_diversity >= self.minimum_yaw_diversity))

    def log(self, value):
        try:
            with open(self.log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(value, sort_keys=True) + "\n")
        except OSError:
            pass

    def _select_observation_cluster(self, x, y, z, room_id):
        """Return nearest plausible cluster and whether it crosses a room ID."""
        same_room = [c for c in self.clusters
                     if str(c.get("room_id") or "") == room_id]
        same_cluster = min(
            same_room,
            key=lambda c: math.hypot(c["x"] - x, c["y"] - y),
            default=None)
        same_distance = (
            math.hypot(same_cluster["x"] - x, same_cluster["y"] - y)
            if same_cluster is not None else math.inf)
        cross_room = [
            c for c in self.clusters
            if str(c.get("room_id") or "") != room_id and
            abs(float(c.get("z", 0.0)) - z) <=
            self.cross_room_duplicate_max_vertical and
            math.hypot(c["x"] - x, c["y"] - y) <=
            self.cross_room_duplicate_radius]
        cross_cluster = min(
            cross_room,
            key=lambda c: math.hypot(c["x"] - x, c["y"] - y),
            default=None)
        cross_distance = (
            math.hypot(cross_cluster["x"] - x, cross_cluster["y"] - y)
            if cross_cluster is not None else math.inf)
        # Prefer an already confirmed physical source over a farther pending
        # room-local fragment. Otherwise ordinary same-room association wins.
        use_cross = bool(
            cross_cluster is not None and
            (same_cluster is None or same_distance > self.radius or
             (cross_cluster.get("confirmed", False) and
              not same_cluster.get("confirmed", False) and
              cross_distance + 0.10 < same_distance)))
        if use_cross:
            return cross_cluster, cross_distance, True
        return same_cluster, same_distance, False

    def on_observation(self, msg):
        try:
            obs = json.loads(msg.data); p = obs["map_position"]
            x, y, z = float(p["x"]), float(p["y"]), float(p.get("z", 0.0))
        except (ValueError, TypeError, KeyError):
            return
        room_id = str(obs.get("room_id") or "")
        stamp = float(obs.get("timestamp", time.time()))
        cluster, distance, cross_room_duplicate = (
            self._select_observation_cluster(x, y, z, room_id))
        pixel = obs.get("pixel", {}) if isinstance(obs.get("pixel"), dict) else {}
        strict_continuity = bool(
            not cross_room_duplicate and cluster is not None and not cluster.get("confirmed", False) and
            str(pixel.get("shape_gate", "strict")) == "strict" and
            not bool(obs.get("profile_relaxed", False)) and
            float(obs.get("confidence", 0.0)) >= 0.92 and
            stamp - float(cluster.get("last_observation_stamp", -math.inf))
            <= self.continuity_seconds and
            distance <= self.continuity_radius)
        if cluster is None or (distance > self.radius and
                               not strict_continuity):
            cluster = {"id": "hazard-%02d" % self.next_id, "x": x, "y": y, "z": z, "hits": 0,
                       "confidence": 0.0, "confirmed": False, "room_id": obs.get("room_id"), "verify_sent": False,
                       "confirmed_stamp": None, "views": [], "representative": None,
                       "representative_quality": -math.inf,
                       "profile_relaxed_hits": 0, "strict_hits": 0,
                       "strict_pair_hits": 0, "strict_pair_views": [],
                       "strict_pair_stamps": [],
                       "observation_stamps": [],
                       "last_observation_stamp": -math.inf}
            self.next_id += 1; self.clusters.append(cluster)
        n = cluster["hits"]
        cluster["x"] = (cluster["x"] * n + x) / (n + 1); cluster["y"] = (cluster["y"] * n + y) / (n + 1); cluster["z"] = (cluster["z"] * n + z) / (n + 1)
        cluster["hits"] += 1; cluster["confidence"] = max(cluster["confidence"], float(obs.get("confidence", 0.0)))
        cluster["last_observation_stamp"] = stamp
        cluster["observation_stamps"].append(stamp)
        # An oblique contour may pass the detector's geometric fallback, but
        # it is not equivalent to a fully visible strict sphere.  Treat it
        # as relaxed confirmation evidence here: otherwise three skewed
        # views of a red box/furniture edge can be promoted as a hazard.
        # A genuine distant ball remains eligible through the existing
        # five-view + 15-degree parallax route, with no motion added.
        # A high-confidence YOLOv8 red_ball box is the learned detector's
        # strict class decision.  Do not route it through the legacy
        # HSV-oblique five-hit fallback merely because its diagnostic
        # ``shape_gate`` is named ``yolo``.  It still requires the ordinary
        # three observations and independent view baseline/yaw diversity.
        neural_strict = str(obs.get("source", "")) == "yolov8_rgbd"
        relaxed_shape = (str(pixel.get("shape_gate", "strict")) != "strict" and
                         not neural_strict)
        if bool(obs.get("profile_relaxed", False)) or relaxed_shape:
            cluster["profile_relaxed_hits"] += 1
        else:
            cluster["strict_hits"] += 1
        # Prefer a confident, nearer, horizontally centred sphere: this is
        # the least sensitive RGB-D ray to depth and bearing error.
        depth_m = float(obs.get("depth_m", 0.0))
        # Depth equal to the configured 8 m range is a clipped ray, not a
        # precise surface return.  It used to beat an accurate 6.6 m D0 view
        # merely because the ball was centred in the image, shifting the
        # published representative by nearly a metre.
        clipped_depth_penalty = 0.12 if depth_m >= 7.8 else 0.0
        quality = (float(obs.get("confidence", 0.0))
                   - 0.012 * depth_m
                   - 0.00020 * abs(float(pixel.get("u", 320.0)) - 320.0)
                   - clipped_depth_penalty)
        if quality > cluster["representative_quality"]:
            cluster["representative_quality"] = quality
            cluster["representative"] = {
                "x": x, "y": y, "z": z,
                "timestamp": float(obs.get("timestamp", time.time())),
                "source": str(obs.get("source", "rgbd_active_search"))}
        camera = obs.get("camera_position", {})
        try:
            view = (float(camera["x"]), float(camera["y"]), float(camera["yaw"]))
            if not cluster["views"] or any(
                    math.hypot(view[0] - item[0], view[1] - item[1]) > .04 or
                    abs((view[2] - item[2] + math.pi) % (2*math.pi) - math.pi) > math.radians(2)
                    for item in cluster["views"]):
                cluster["views"].append(view)
            if self.strict_pair_observation_eligible(obs, pixel):
                cluster["strict_pair_hits"] += 1
                cluster["strict_pair_stamps"].append(stamp)
                if not cluster["strict_pair_views"] or any(
                        math.hypot(view[0] - item[0], view[1] - item[1]) > .04 or
                        abs((view[2] - item[2] + math.pi) %
                            (2 * math.pi) - math.pi) > math.radians(2)
                        for item in cluster["strict_pair_views"]):
                    cluster["strict_pair_views"].append(view)
        except (KeyError, TypeError, ValueError):
            pass
        just_confirmed = not cluster["confirmed"] and self.is_confirmed(cluster)
        cluster["confirmed"] = self.is_confirmed(cluster)
        if just_confirmed:
            cluster["confirmed_stamp"] = float(obs.get("timestamp", time.time()))
        baseline, yaw_diversity = self.view_evidence(cluster)
        event = {"timestamp": time.time(), "event": "confirmed" if just_confirmed else "observation",
                 "candidate": self.public(cluster), "observation": obs,
                 "observation_room_id": room_id,
                 "cross_room_duplicate_association": bool(
                     cross_room_duplicate),
                 "view_baseline_m": round(baseline, 3),
                 "view_yaw_diversity_deg": round(math.degrees(yaw_diversity), 2)}
        self.log(event); self.persist()
        canonical_room_id = str(cluster.get("room_id") or "")
        self.publish_room_status(canonical_room_id)
        if room_id and room_id != canonical_room_id:
            self.publish_room_status(room_id)
        # One bounded verification request for an unconfirmed low/medium confidence candidate.
        if (self.enable_verify_goal and not cluster["confirmed"] and
                not cluster["verify_sent"] and cluster["hits"] >= 1):
            cluster["verify_sent"] = True
            self.verify_pub.publish(String(data=json.dumps({"candidate_id": cluster["id"], "position": self.public(cluster)["position"], "room_id": cluster.get("room_id"), "standoff_min_m": 1.5, "standoff_max_m": 3.0})))


if __name__ == "__main__":
    rospy.init_node("hazard_candidate_tracker")
    HazardTracker(); rospy.spin()
