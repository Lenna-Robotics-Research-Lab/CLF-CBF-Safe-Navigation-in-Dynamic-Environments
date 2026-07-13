#!/usr/bin/env python3

import rospy
import math

from gazebo_msgs.msg import ModelState
from gazebo_msgs.srv import SetModelState


def main():

    rospy.init_node("move_cylinder")

    rospy.wait_for_service('/gazebo/set_model_state')

    set_state = rospy.ServiceProxy(
        '/gazebo/set_model_state',
        SetModelState)

    rate = rospy.Rate(30)

    t = 0.0

    while not rospy.is_shutdown():

        state = ModelState()

        state.model_name = "moving_cylinder"

        # Back-and-forth motion
        state.pose.position.x = 1.0 + math.sin(t)

        state.pose.position.y = -0.5

        state.pose.position.z = 0.0

        state.pose.orientation.w = 1.0

        try:
            set_state(state)
        except:
            pass

        t += 0.03

        rate.sleep()


if __name__ == "__main__":
    main()