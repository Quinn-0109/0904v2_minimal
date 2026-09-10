#!/usr/bin/env python3
"""Follow TARE waypoints with world-frame planar velocity commands."""
import json
import math

import rospy
from geometry_msgs.msg import PointStamped, Twist
from nav_msgs.msg import Odometry, Path


def clamp(value, low, high):
    return max(low, min(high, value))


def compute_world_cmd(robot_x, robot_y, target_x, target_y, k_v, v_max, goal_tol):
    dx = target_x - robot_x
    dy = target_y - robot_y
    dist = math.hypot(dx, dy)
    if dist <= goal_tol:
        return 0.0, 0.0, True
    speed = min(v_max, k_v * dist)
    scale = speed / max(dist, 1e-6)
    return dx * scale, dy * scale, False


def select_active_waypoint(active_wp, new_wp, robot_x, robot_y, now, goal_tol,
                           latch_timeout, latch_enabled):
    """Keep chasing the current waypoint until it is reached or expires."""
    if not latch_enabled:
        return (new_wp if new_wp is not None else active_wp), False
    if active_wp is None:
        return new_wp, False

    dx = active_wp[0] - robot_x
    dy = active_wp[1] - robot_y
    if math.hypot(dx, dy) <= goal_tol:
        return new_wp, True
    if now - active_wp[3] >= latch_timeout:
        return new_wp, False
    return active_wp, False


def path_preference_allowed(active_wp, robot_x, robot_y, now, goal_tol,
                            latch_timeout, latch_enabled):
    if not latch_enabled or active_wp is None:
        return True
    if math.hypot(active_wp[0] - robot_x, active_wp[1] - robot_y) <= goal_tol:
        return True
    return now - active_wp[3] >= latch_timeout


def path_length(path):
    total = 0.0
    for prev, cur in zip(path, path[1:]):
        total += math.hypot(cur[0] - prev[0], cur[1] - prev[1])
    return total


def select_path_target(robot_x, robot_y, path, lookahead, anchor_tolerance, min_path_length=0.0):
    """Pick a forward target on a path whose first pose is normally the robot."""
    if not path:
        return None
    if path_length(path) < min_path_length:
        return None
    start_dist = math.hypot(path[0][0] - robot_x, path[0][1] - robot_y)
    if start_dist <= anchor_tolerance:
        start_index = 0
    else:
        start_index = min(
            range(len(path)),
            key=lambda i: math.hypot(path[i][0] - robot_x, path[i][1] - robot_y),
        )

    travelled = 0.0
    prev = (robot_x, robot_y)
    for point in path[start_index:]:
        travelled += math.hypot(point[0] - prev[0], point[1] - prev[1])
        if travelled >= lookahead:
            return point
        prev = (point[0], point[1])
    return path[-1]


def path_msg_to_points(msg):
    return [
        (pose.pose.position.x, pose.pose.position.y, pose.pose.position.z)
        for pose in msg.poses
    ]


def parse_layout_topology(metadata):
    floors = metadata.get("floors", [])
    if not floors:
        return None
    floor = floors[0]
    lobby = floor.get("lobby_bounds")
    corridor = floor.get("corridor_bounds")
    rooms = []
    if not corridor:
        return None
    regions = []
    if lobby:
        regions.append({
            "id": "lobby",
            "bounds": {
                "x_min": float(lobby["x_min"]),
                "x_max": float(lobby["x_max"]),
                "y_min": float(lobby["y_min"]),
                "y_max": float(lobby["y_max"]),
            },
        })
    regions.append({
        "id": "corridor",
        "bounds": {
            "x_min": float(corridor["x_min"]),
            "x_max": float(corridor["x_max"]),
            "y_min": float(corridor["y_min"]),
            "y_max": float(corridor["y_max"]),
        },
    })
    for room in floor.get("rooms", []):
        bounds = room.get("bounds")
        door_pose = room.get("door_pose")
        if not bounds or not door_pose:
            continue
        room_info = {
            "id": room.get("id", ""),
            "bounds": {
                "x_min": float(bounds["x_min"]),
                "x_max": float(bounds["x_max"]),
                "y_min": float(bounds["y_min"]),
                "y_max": float(bounds["y_max"]),
            },
            "door": (float(door_pose[0]), float(door_pose[1])),
        }
        rooms.append(room_info)
        regions.append({"id": room_info["id"], "bounds": room_info["bounds"]})
    if not rooms:
        return None
    return {
        "lobby": {
            "x_min": float(lobby["x_min"]),
            "x_max": float(lobby["x_max"]),
            "y_min": float(lobby["y_min"]),
            "y_max": float(lobby["y_max"]),
        } if lobby else None,
        "corridor": {
            "x_min": float(corridor["x_min"]),
            "x_max": float(corridor["x_max"]),
            "y_min": float(corridor["y_min"]),
            "y_max": float(corridor["y_max"]),
        },
        "rooms": rooms,
        "regions": regions,
    }


def load_layout_topology(path):
    if not path:
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return parse_layout_topology(json.load(f))
    except (OSError, ValueError, KeyError) as exc:
        rospy.logwarn("waypoint_follower_world: topology guard disabled, failed to load %s: %s", path, exc)
        return None


def _inside_bounds(x, y, bounds, margin=0.0):
    return (
        float(bounds["x_min"]) - margin <= x <= float(bounds["x_max"]) + margin and
        float(bounds["y_min"]) - margin <= y <= float(bounds["y_max"]) + margin
    )


def clamp_target_to_layout(target, topology, inset=0.05):
    if target is None or topology is None:
        return target, False
    x, y = target[0], target[1]
    z = target[2] if len(target) > 2 else 0.0
    regions = topology.get("regions", [])
    if any(_inside_bounds(x, y, region["bounds"]) for region in regions):
        return target, False
    best = None
    best_dist = None
    for region in regions:
        bounds = region["bounds"]
        cx = clamp(x, bounds["x_min"] + inset, bounds["x_max"] - inset)
        cy = clamp(y, bounds["y_min"] + inset, bounds["y_max"] - inset)
        dist = math.hypot(cx - x, cy - y)
        if best is None or dist < best_dist:
            best = (cx, cy, z) + tuple(target[3:])
            best_dist = dist
    return (best if best is not None else target), best is not None


def robot_in_any_room(robot_x, robot_y, topology):
    if not topology:
        return False
    return any(_inside_bounds(robot_x, robot_y, room["bounds"]) for room in topology["rooms"])


def room_containing_point(x, y, topology):
    if not topology:
        return None
    for room in topology["rooms"]:
        if _inside_bounds(x, y, room["bounds"]):
            return room
    return None


def room_containing_robot(x, y, topology):
    return room_containing_point(x, y, topology)


def room_by_id(topology, room_id):
    if not topology:
        return None
    for room in topology["rooms"]:
        if room.get("id") == room_id:
            return room
    return None


def topology_region_id(point, topology):
    if point is None or topology is None:
        return None
    clamped, _ = clamp_target_to_layout(point, topology)
    x, y = clamped[0], clamped[1]
    room = room_containing_point(x, y, topology)
    if room is not None:
        return room.get("id")
    lobby = topology.get("lobby")
    if lobby and _inside_bounds(x, y, lobby):
        return "lobby"
    if _inside_bounds(x, y, topology["corridor"]):
        return "corridor"
    return None


def should_replace_latched_waypoint(active_wp, new_wp, topology):
    """Let a fresh TARE target in another region override stale latched intent."""
    active_region = topology_region_id(active_wp, topology)
    new_region = topology_region_id(new_wp, topology)
    if active_region is None or new_region is None:
        return False
    if not str(new_region).startswith("floor_"):
        return False
    return active_region != new_region


def select_topology_intent_target(path_target, latest_wp, latest_is_fresh, topology):
    """Use TARE's waypoint as room intent when the path lookahead is only a corridor step."""
    if topology is None or path_target is None:
        return path_target
    path_room = room_containing_point(path_target[0], path_target[1], topology)
    if latest_is_fresh and latest_wp is not None:
        waypoint_room = room_containing_point(latest_wp[0], latest_wp[1], topology)
        if waypoint_room is not None and (
                path_room is None or waypoint_room.get("id") != path_room.get("id")):
            return latest_wp
    if path_room is not None:
        return path_target
    return path_target


def room_entry_target(room, topology, room_entry_depth):
    corridor = topology["corridor"]
    bounds = room["bounds"]
    door_y = room["door"][1]
    if bounds["x_min"] >= corridor["x_max"]:
        entry_x = min(bounds["x_min"] + room_entry_depth, bounds["x_max"])
    else:
        entry_x = max(bounds["x_max"] - room_entry_depth, bounds["x_min"])
    return (entry_x, door_y, 0.0)


def select_room_route_target(robot_x, robot_y, topology, door_tolerance, room_entry_depth, target=None):
    corridor = topology["corridor"]
    corridor_center_x = 0.5 * (corridor["x_min"] + corridor["x_max"])
    room = room_containing_point(target[0], target[1], topology) if target is not None else None
    if room is None:
        if target is not None and _inside_bounds(target[0], target[1], corridor):
            return target
        rooms = sorted(
            topology["rooms"],
            key=lambda candidate: (max(0.0, candidate["door"][1] - robot_y), abs(candidate["door"][1] - robot_y)),
        )
        room = rooms[0]
    door_x, door_y = room["door"]
    if robot_y < door_y - door_tolerance:
        return (corridor_center_x, door_y, 0.0)

    return room_entry_target(room, topology, room_entry_depth)


def apply_room_entry_commit(robot_x, robot_y, target, topology, commit,
                            door_tolerance, room_entry_depth):
    """Commit to a selected room until its interior entry point is reached."""
    if target is None or topology is None:
        return target, commit, False

    def route_to_committed_room(room):
        entry = room_entry_target(room, topology, room_entry_depth)
        if math.hypot(robot_x - entry[0], robot_y - entry[1]) <= door_tolerance:
            return None
        door_y = room["door"][1]
        if robot_y < door_y - door_tolerance:
            corridor = topology["corridor"]
            corridor_center_x = 0.5 * (corridor["x_min"] + corridor["x_max"])
            return (corridor_center_x, door_y, entry[2])
        return entry

    if commit is not None:
        room = room_by_id(topology, commit.get("room_id"))
        if room is None:
            return target, None, False
        routed = route_to_committed_room(room)
        if routed is None:
            return target, None, False
        return routed, commit, True

    if robot_in_any_room(robot_x, robot_y, topology):
        return target, None, False

    corridor = topology["corridor"]
    lobby = topology.get("lobby")
    if not (_inside_bounds(robot_x, robot_y, corridor, margin=0.1) or
            (lobby and _inside_bounds(robot_x, robot_y, lobby, margin=0.1))):
        return target, None, False

    room = room_containing_point(target[0], target[1], topology)
    if room is None:
        return target, None, False

    new_commit = {"room_id": room.get("id", "")}
    routed = route_to_committed_room(room)
    if routed is None:
        return target, None, False
    return routed, new_commit, True


def global_path_needs_corridor(global_path, topology, margin=0.25):
    if not global_path or not topology or not topology.get("lobby"):
        return False
    lobby = topology["lobby"]
    return any(point[1] > lobby["y_max"] + margin for point in global_path)


def select_corridor_entry_target(topology, room_entry_depth, z=0.0):
    corridor = topology["corridor"]
    corridor_center_x = 0.5 * (corridor["x_min"] + corridor["x_max"])
    entry_y = min(corridor["y_min"] + room_entry_depth, corridor["y_max"])
    return (corridor_center_x, entry_y, z)


def build_lobby_bootstrap_targets(topology, room_entry_depth, z=0.0):
    if topology is None or not topology.get("lobby"):
        return []
    lobby = topology["lobby"]
    x_min = lobby["x_min"]
    x_max = lobby["x_max"]
    y_min = lobby["y_min"]
    y_max = lobby["y_max"]
    sweep_y = max(y_min + 1.0, y_max - 1.5)
    sweep_y = min(sweep_y, y_max - 0.5)
    left_x = min(0.0, x_min + 2.0)
    right_x = max(0.0, x_max - 2.0)
    return [
        (left_x, sweep_y, z),
        (right_x, sweep_y, z),
        select_corridor_entry_target(topology, room_entry_depth, z),
    ]


def select_lobby_bootstrap_target(robot_x, robot_y, target, topology, state,
                                  global_path, room_entry_depth, goal_tol):
    if topology is None or not topology.get("lobby") or target is None:
        return target, False
    if state.get("done"):
        return target, False
    lobby = topology["lobby"]
    if not state.get("active"):
        if not _inside_bounds(robot_x, robot_y, lobby):
            return target, False
        state["targets"] = build_lobby_bootstrap_targets(
            topology, room_entry_depth, target[2] if len(target) > 2 else 0.0,
        )
        state["index"] = 0
        state["active"] = True
    targets = state.get("targets", [])
    while state.get("index", 0) < len(targets):
        current = targets[state["index"]]
        if math.hypot(robot_x - current[0], robot_y - current[1]) > goal_tol:
            return current, True
        state["index"] += 1
    state["active"] = False
    state["done"] = True
    return target, False


def apply_topology_guard(robot_x, robot_y, target, topology, room_entered,
                         door_tolerance, corridor_margin, room_entry_depth,
                         global_path=None):
    if target is None or topology is None:
        return target, room_entered, False
    robot_room = room_containing_robot(robot_x, robot_y, topology)
    if robot_room is not None:
        target_room = room_containing_point(target[0], target[1], topology)
        if target_room is None or target_room.get("id") != robot_room.get("id"):
            door_x, door_y = robot_room["door"]
            corridor = topology["corridor"]
            corridor_center_x = 0.5 * (corridor["x_min"] + corridor["x_max"])
            return (corridor_center_x, door_y, target[2] if len(target) > 2 else 0.0), True, True
        return target, True, False
    lobby = topology.get("lobby")
    if lobby and _inside_bounds(robot_x, robot_y, lobby) and global_path_needs_corridor(global_path, topology):
        z = target[2] if len(target) > 2 else 0.0
        return select_corridor_entry_target(topology, room_entry_depth, z), room_entered, True
    corridor = topology["corridor"]
    if not _inside_bounds(robot_x, robot_y, corridor, corridor_margin):
        return target, room_entered, False
    if lobby and _inside_bounds(target[0], target[1], lobby):
        return target, False, False
    target_room = room_containing_point(target[0], target[1], topology)
    if target_room is None and room_entered:
        return target, False, False
    guarded_target = select_room_route_target(
        robot_x, robot_y, topology, door_tolerance, room_entry_depth, target=target,
    )
    if guarded_target == target:
        return target, False, False
    return guarded_target, False, True


def route_topology_target(robot_x, robot_y, target, topology, room_entered, commit,
                          door_tolerance, corridor_margin, room_entry_depth,
                          global_path=None):
    if target is None or topology is None:
        return target, room_entered, commit, False
    target, clamped = clamp_target_to_layout(target, topology)
    target_room = room_containing_point(target[0], target[1], topology)
    robot_room = room_containing_robot(robot_x, robot_y, topology)
    if commit is not None and target_room is not None and commit.get("room_id") != target_room.get("id"):
        commit = {"room_id": target_room.get("id", "")}
    if commit is None and target_room is not None and robot_room is not None:
        if target_room.get("id") != robot_room.get("id"):
            commit = {"room_id": target_room.get("id", "")}
    target, commit, committed = apply_room_entry_commit(
        robot_x, robot_y, target, topology, commit,
        door_tolerance, room_entry_depth,
    )
    target, room_entered, guarded = apply_topology_guard(
        robot_x, robot_y, target, topology, room_entered,
        door_tolerance, corridor_margin, room_entry_depth,
        global_path=global_path,
    )
    return target, room_entered, commit, clamped or committed or guarded


def main():
    rospy.init_node("waypoint_follower_world")
    waypoint_topic = rospy.get_param("~waypoint", "/way_point")
    path_topic = rospy.get_param("~path", "/sensor_coverage_planner/global_path_full")
    odom_topic = rospy.get_param("~odom", "/Odometry_gazebo")
    cmd_topic = rospy.get_param("~cmd_vel", "/cmd_vel")
    k_v = float(rospy.get_param("~k_v", 0.8))
    v_max = float(rospy.get_param("~v_max", 1.0))
    goal_tol = float(rospy.get_param("~goal_tol", 0.35))
    wp_timeout = float(rospy.get_param("~wp_timeout", 3.0))
    latch_enabled = bool(rospy.get_param("~latch_waypoint", True))
    latch_timeout = float(rospy.get_param("~latch_timeout", 20.0))
    prefer_path = bool(rospy.get_param("~prefer_path", True))
    path_timeout = float(rospy.get_param("~path_timeout", 3.0))
    path_lookahead = float(rospy.get_param("~path_lookahead", 2.0))
    path_anchor_tolerance = float(rospy.get_param("~path_anchor_tolerance", 3.0))
    path_min_length = float(rospy.get_param("~path_min_length", 1.0))
    topology_guard = bool(rospy.get_param("~topology_guard", False))
    layout_metadata = rospy.get_param("~layout_metadata", "")
    topology = load_layout_topology(layout_metadata) if topology_guard else None
    topology_door_tolerance = float(rospy.get_param("~topology_door_tolerance", 0.75))
    topology_corridor_margin = float(rospy.get_param("~topology_corridor_margin", 0.5))
    topology_room_entry_depth = float(rospy.get_param("~topology_room_entry_depth", 2.0))

    state = {
        "odom": None,
        "latest_wp": None,
        "active_wp": None,
        "wp_time": rospy.Time(0),
        "latest_path": [],
        "path_time": rospy.Time(0),
        "room_entered": False,
        "room_entry_commit": None,
        "lobby_bootstrap": {"active": False, "done": False, "index": 0},
    }

    def on_odom(msg):
        p = msg.pose.pose.position
        state["odom"] = (p.x, p.y)

    def on_waypoint(msg):
        now = rospy.Time.now()
        state["latest_wp"] = (msg.point.x, msg.point.y, msg.point.z, now.to_sec())
        state["wp_time"] = now

    def on_path(msg):
        state["latest_path"] = path_msg_to_points(msg)
        state["path_time"] = rospy.Time.now()

    rospy.Subscriber(odom_topic, Odometry, on_odom, queue_size=50)
    rospy.Subscriber(waypoint_topic, PointStamped, on_waypoint, queue_size=10)
    if (prefer_path or topology_guard) and path_topic:
        rospy.Subscriber(path_topic, Path, on_path, queue_size=5)
    pub = rospy.Publisher(cmd_topic, Twist, queue_size=1)
    rate = rospy.Rate(20)
    rospy.loginfo("waypoint_follower_world: waypoint=%s path=%s odom=%s -> %s",
                  waypoint_topic, path_topic if prefer_path else "(disabled)", odom_topic, cmd_topic)

    while not rospy.is_shutdown():
        cmd = Twist()
        odom = state["odom"]
        latest_wp = state["latest_wp"]
        latest_is_fresh = (rospy.Time.now() - state["wp_time"]).to_sec() <= wp_timeout
        latest_path_is_fresh = (rospy.Time.now() - state["path_time"]).to_sec() <= path_timeout
        handled_by_path = False
        now_sec = rospy.Time.now().to_sec()
        if (odom is not None and (latest_wp is not None or state["active_wp"] is not None)
                and latest_is_fresh):
            if should_replace_latched_waypoint(state["active_wp"], latest_wp, topology):
                state["active_wp"] = None
                state["room_entry_commit"] = None
                state["room_entered"] = False
            active_wp, reached = select_active_waypoint(
                state["active_wp"], latest_wp, odom[0], odom[1], now_sec,
                goal_tol, latch_timeout, latch_enabled,
            )
            state["active_wp"] = active_wp
            if reached:
                rospy.loginfo("waypoint_follower_world: reached latched waypoint")
        path_allowed = odom is not None and path_preference_allowed(
            state["active_wp"], odom[0], odom[1], now_sec,
            goal_tol, latch_timeout, latch_enabled,
        )
        if odom is not None and path_allowed and prefer_path and state["latest_path"] and latest_path_is_fresh:
            target = select_path_target(
                odom[0], odom[1], state["latest_path"],
                path_lookahead, path_anchor_tolerance, path_min_length,
        )
            if target is not None:
                intent_target = select_topology_intent_target(
                    target, latest_wp, latest_is_fresh, topology,
                )
                target, state["room_entered"], state["room_entry_commit"], overridden = route_topology_target(
                    odom[0], odom[1], intent_target, topology, state["room_entered"],
                    state["room_entry_commit"],
                    topology_door_tolerance, topology_corridor_margin, topology_room_entry_depth,
                    global_path=state["latest_path"],
                )
                target, bootstrapped = select_lobby_bootstrap_target(
                    odom[0], odom[1], target, topology, state["lobby_bootstrap"],
                    state["latest_path"], topology_room_entry_depth, goal_tol,
                )
                overridden = overridden or bootstrapped
                if overridden:
                    rospy.loginfo_throttle(2.0, "waypoint_follower_world: topology target=(%.2f, %.2f)",
                                           target[0], target[1])
                target, clamped = clamp_target_to_layout(target, topology)
                if clamped:
                    rospy.loginfo_throttle(2.0, "waypoint_follower_world: clamped target=(%.2f, %.2f)",
                                           target[0], target[1])
                vx, vy, done = compute_world_cmd(odom[0], odom[1], target[0], target[1], k_v, v_max, goal_tol)
                if not done:
                    cmd.linear.x = vx
                    cmd.linear.y = vy
                handled_by_path = not done
        if (not handled_by_path and odom is not None and
                (latest_wp is not None or state["active_wp"] is not None) and latest_is_fresh):
            active_wp = state["active_wp"]
            if active_wp is None:
                pub.publish(cmd)
                rate.sleep()
                continue
            target, state["room_entered"], state["room_entry_commit"], overridden = route_topology_target(
                odom[0], odom[1], active_wp, topology, state["room_entered"],
                state["room_entry_commit"],
                topology_door_tolerance, topology_corridor_margin, topology_room_entry_depth,
                global_path=state["latest_path"],
            )
            target, bootstrapped = select_lobby_bootstrap_target(
                odom[0], odom[1], target, topology, state["lobby_bootstrap"],
                state["latest_path"], topology_room_entry_depth, goal_tol,
            )
            overridden = overridden or bootstrapped
            if overridden:
                rospy.loginfo_throttle(2.0, "waypoint_follower_world: topology target=(%.2f, %.2f)",
                                       target[0], target[1])
            target, clamped = clamp_target_to_layout(target, topology)
            if clamped:
                rospy.loginfo_throttle(2.0, "waypoint_follower_world: clamped target=(%.2f, %.2f)",
                                       target[0], target[1])
            vx, vy, done = compute_world_cmd(odom[0], odom[1], target[0], target[1], k_v, v_max, goal_tol)
            if not done:
                cmd.linear.x = vx
                cmd.linear.y = vy
        pub.publish(cmd)
        rate.sleep()


if __name__ == "__main__":
    main()
