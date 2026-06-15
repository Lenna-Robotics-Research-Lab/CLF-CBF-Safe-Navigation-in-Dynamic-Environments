#!/usr/bin/env python3

import rospy
from gazebo_msgs.msg import ModelStates
from gazebo_msgs.msg import ModelState
from gazebo_msgs.srv import SetModelState


class PersonFollower:

    def __init__(self):

        rospy.init_node("move_cylinder_follow_person")

        rospy.wait_for_service('/gazebo/set_model_state')
        self.set_state = rospy.ServiceProxy('/gazebo/set_model_state', SetModelState)

        rospy.Subscriber('/gazebo/model_states', ModelStates, self.callback)

        self.person_pose = None

        # cylinder geometry assumption (adjust if needed)
        self.cylinder_height = 1.0
        self.z_offset = self.cylinder_height / 2.0  # keeps bottom on actor point

        self.rate = rospy.Rate(30)

    def callback(self, msg):

        try:
            idx = msg.name.index("person1")
            self.person_pose = msg.pose[idx]
        except ValueError:
            self.person_pose = None

    def run(self):

        while not rospy.is_shutdown():

            if self.person_pose is not None:

                state = ModelState()
                state.model_name = "moving_cylinder"

                # Copy XY from actor
                state.pose.position.x = self.person_pose.position.x
                state.pose.position.y = self.person_pose.position.y

                # Add vertical bias so cylinder sits above ground
                state.pose.position.z = 0.0

                # Keep fixed upright orientation (no tilting with actor)
                state.pose.orientation.x = 0.0
                state.pose.orientation.y = 0.0
                state.pose.orientation.z = 0.0
                state.pose.orientation.w = 1.0

                try:
                    self.set_state(state)
                except:
                    pass

            self.rate.sleep()


if __name__ == "__main__":
    node = PersonFollower()
    node.run()