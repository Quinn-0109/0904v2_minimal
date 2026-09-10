#!/usr/bin/env python3
"""Publish SimEnv layout bounds as TARE navigation/coverage polygons."""
import json


def load_metadata(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _iter_bounds(metadata):
    for floor in metadata.get("floors", []):
        for key in ("lobby_bounds", "corridor_bounds", "elevator_bounds"):
            bounds = floor.get(key)
            if bounds:
                yield bounds
        for room in floor.get("rooms", []):
            bounds = room.get("bounds")
            if bounds:
                yield bounds


def compute_outer_bounds(metadata, margin=0.0):
    bounds_list = list(_iter_bounds(metadata))
    if not bounds_list:
        raise ValueError("layout metadata does not contain any floor bounds")

    x_min = min(float(b["x_min"]) for b in bounds_list) - margin
    x_max = max(float(b["x_max"]) for b in bounds_list) + margin
    y_min = min(float(b["y_min"]) for b in bounds_list) - margin
    y_max = max(float(b["y_max"]) for b in bounds_list) + margin
    return (round(x_min, 6), round(x_max, 6), round(y_min, 6), round(y_max, 6))


def compute_nogo_bounds(metadata, margin=0.0):
    nogo_bounds = []
    for floor in metadata.get("floors", []):
        bounds = floor.get("elevator_bounds")
        if not bounds:
            continue
        nogo_bounds.append((
            round(float(bounds["x_min"]) - margin, 6),
            round(float(bounds["x_max"]) + margin, 6),
            round(float(bounds["y_min"]) - margin, 6),
            round(float(bounds["y_max"]) + margin, 6),
        ))
    return nogo_bounds


def make_polygon(bounds, frame_id):
    from geometry_msgs.msg import Point32, PolygonStamped

    x_min, x_max, y_min, y_max = bounds
    msg = PolygonStamped()
    msg.header.frame_id = frame_id
    msg.polygon.points = [
        Point32(x=x_min, y=y_min, z=0.0),
        Point32(x=x_max, y=y_min, z=0.0),
        Point32(x=x_max, y=y_max, z=0.0),
        Point32(x=x_min, y=y_max, z=0.0),
    ]
    return msg


def make_nogo_polygon(bounds_list, frame_id):
    from geometry_msgs.msg import Point32, PolygonStamped

    msg = PolygonStamped()
    msg.header.frame_id = frame_id
    for polygon_id, bounds in enumerate(bounds_list):
        x_min, x_max, y_min, y_max = bounds
        z = float(polygon_id)
        msg.polygon.points.extend([
            Point32(x=x_min, y=y_min, z=z),
            Point32(x=x_max, y=y_min, z=z),
            Point32(x=x_max, y=y_max, z=z),
            Point32(x=x_min, y=y_max, z=z),
        ])
    return msg


def main():
    import rospy
    from geometry_msgs.msg import PolygonStamped

    rospy.init_node("layout_boundary_publisher")
    metadata_path = rospy.get_param("~metadata", "/workspace/SimEnv/generated_building/layout_metadata.json")
    frame_id = rospy.get_param("~frame", "map")
    margin = float(rospy.get_param("~margin", 0.5))
    nogo_margin = float(rospy.get_param("~nogo_margin", 0.2))
    nav_topic = rospy.get_param("~navigation_topic", "/navigation_boundary")
    coverage_topic = rospy.get_param("~coverage_topic", "/sensor_coverage_planner/coverage_boundary")
    nogo_topic = rospy.get_param("~nogo_topic", "/sensor_coverage_planner/nogo_boundary")
    rate_hz = float(rospy.get_param("~rate", 1.0))

    metadata = load_metadata(metadata_path)
    bounds = compute_outer_bounds(metadata, margin=margin)
    polygon = make_polygon(bounds, frame_id)
    nogo_bounds = compute_nogo_bounds(metadata, margin=nogo_margin)
    nogo_polygon = make_nogo_polygon(nogo_bounds, frame_id)
    nav_pub = rospy.Publisher(nav_topic, PolygonStamped, queue_size=1, latch=True)
    coverage_pub = rospy.Publisher(coverage_topic, PolygonStamped, queue_size=1, latch=True)
    nogo_pub = rospy.Publisher(nogo_topic, PolygonStamped, queue_size=1, latch=True)

    rospy.loginfo("layout_boundary_publisher: bounds=%s nogo=%s frame=%s", bounds, nogo_bounds, frame_id)
    rate = rospy.Rate(rate_hz)
    while not rospy.is_shutdown():
        polygon.header.stamp = rospy.Time.now()
        nogo_polygon.header.stamp = polygon.header.stamp
        nav_pub.publish(polygon)
        coverage_pub.publish(polygon)
        if nogo_polygon.polygon.points:
            nogo_pub.publish(nogo_polygon)
        rate.sleep()


if __name__ == "__main__":
    main()
