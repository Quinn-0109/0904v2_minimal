#!/usr/bin/env python3
"""Open the public entrance before manual SLAM driving begins."""

import rospy

from building_generator_interfaces.srv import SetDoorState


def main():
    rospy.init_node("open_mapping_entrance")
    service_name = "/set_door_state"
    door_id = rospy.get_param("~door_id", "main_entrance")
    rospy.loginfo("Waiting to open mapping entrance %s", door_id)
    rospy.wait_for_service(service_name, timeout=60.0)
    response = rospy.ServiceProxy(service_name, SetDoorState)(
        door_id=door_id, open=True
    )
    if not response.accepted:
        raise RuntimeError(response.message)
    rospy.loginfo("Mapping entrance ready: %s", response.message)


if __name__ == "__main__":
    main()
