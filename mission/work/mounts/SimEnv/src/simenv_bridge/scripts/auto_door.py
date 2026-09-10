#!/usr/bin/env python3
"""
auto_door.py — 机器狗靠近门/电梯时自动开门、叫电梯，不干扰探索。

读取 generated_building/door_config.yaml 和 elevator_config.yaml，
拿到每个门/电梯的位置(id, pose[x,y])。
定时用机器狗当前位姿(来自 /Odometry_gazebo，frame=odom==map)测距:
  - 距门 < door_open_dist 且门当前关着 -> /set_door_state(door_id, open=True)
  - 距电梯 < elevator_call_dist -> /call_elevator(elevator_id, 最近楼层, open_doors=True)
  - 离开后(距离 > door_open_dist + hysteresis) 关门(可选，默认不关以免影响)

注意: 门/电梯位置在 yaml 的 pose 前 2 项(x,y)。elevator 门也有自己的
      door_config 条目(kind=elevator)，与电梯本体(elevator_config)分开。

用法: rosrun simenv_bridge auto_door.py
        _door_config:=/workspace/SimEnv/generated_building/door_config.yaml
        _elevator_config:=/workspace/SimEnv/generated_building/elevator_config.yaml
        _odom:=/Odometry_gazebo
"""
import math
import yaml
import rospy
from nav_msgs.msg import Odometry
from building_generator_interfaces.srv import SetDoorState, CallElevator


def _pose_xy(spec):
    """从 door/elevator spec 取世界坐标 (x,y)。pose 是 [x,y,z,r,p,y]。"""
    pose = spec.get("pose") or spec.get("closed_pose") or spec.get("lobby_positions", {}).get("0")
    if pose:
        return float(pose[0]), float(pose[1])
    return None


def main():
    rospy.init_node("auto_door")
    door_cfg = rospy.get_param("~door_config", "/workspace/SimEnv/generated_building/door_config.yaml")
    elev_cfg = rospy.get_param("~elevator_config", "/workspace/SimEnv/generated_building/elevator_config.yaml")
    odom_topic = rospy.get_param("~odom", "/Odometry_gazebo")
    door_open_dist = rospy.get_param("~door_open_dist", 1.8)   # 近于此距离开门
    elev_call_dist = rospy.get_param("~elevator_call_dist", 2.5)
    hysteresis = rospy.get_param("~hysteresis", 0.6)           # 离开后多远才允许再触发
    close_behind = rospy.get_param("~close_behind", False)     # 离开后是否关门

    with open(door_cfg) as f:
        doors = yaml.safe_load(f).get("doors", [])
    with open(elev_cfg) as f:
        elevs = yaml.safe_load(f).get("elevators", [])

    door_specs = []  # [{id, kind, x, y, state}]
    for d in doors:
        xy = _pose_xy(d)
        if xy:
            door_specs.append({"id": d["id"], "kind": d.get("kind", ""),
                               "x": xy[0], "y": xy[1],
                               "state": "open" if d.get("initial_open") else "closed"})
    elev_specs = []  # [{id, x, y, floors, called}]
    for e in elevs:
        lp = e.get("lobby_positions", {}).get("0") or e.get("floor_poses", {}).get("0")
        if lp:
            elev_specs.append({"id": e["id"], "x": float(lp[0]), "y": float(lp[1]),
                               "floors": e.get("served_floors", [0]), "called": False})

    rospy.loginfo("auto_door: loaded %d doors, %d elevators", len(door_specs), len(elev_specs))

    rospy.wait_for_service("/set_door_state", timeout=None)
    rospy.wait_for_service("/call_elevator", timeout=None)
    set_door = rospy.ServiceProxy("/set_door_state", SetDoorState)
    call_elev = rospy.ServiceProxy("/call_elevator", CallElevator)

    robot = {"x": 0.0, "y": 0.0}

    def on_odom(msg):
        robot["x"] = msg.pose.pose.position.x
        robot["y"] = msg.pose.pose.position.y
    rospy.Subscriber(odom_topic, Odometry, on_odom, queue_size=10)

    rate = rospy.Rate(2.0)
    while not rospy.is_shutdown():
        rx, ry = robot["x"], robot["y"]
        for d in door_specs:
            dist = math.hypot(rx - d["x"], ry - d["y"])
            if dist < door_open_dist and d["state"] != "open":
                try:
                    r = set_door(d["id"], True)
                    if r.accepted:
                        d["state"] = "open"
                        rospy.loginfo("auto_door: opened door '%s' (kind=%s, dist=%.2f)",
                                      d["id"], d["kind"], dist)
                except rospy.ServiceException as e:
                    rospy.logwarn("set_door_state failed for %s: %s", d["id"], e)
            elif close_behind and dist > door_open_dist + hysteresis and d["state"] == "open":
                try:
                    r = set_door(d["id"], False)
                    if r.accepted:
                        d["state"] = "closed"
                except rospy.ServiceException:
                    pass
        for e in elev_specs:
            dist = math.hypot(rx - e["x"], ry - e["y"])
            if dist < elev_call_dist and not e["called"]:
                tgt = min(e["floors"]) if e["floors"] else 0
                try:
                    r = call_elev(e["id"], tgt, True)
                    if r.accepted:
                        e["called"] = True
                        rospy.loginfo("auto_door: called elevator '%s' -> floor %d (dist=%.2f)",
                                      e["id"], tgt, dist)
                except rospy.ServiceException as ex:
                    rospy.logwarn("call_elevator failed for %s: %s", e["id"], ex)
            elif dist > elev_call_dist + hysteresis:
                e["called"] = False  # 离开后允许再次呼叫
        rate.sleep()


if __name__ == "__main__":
    main()
