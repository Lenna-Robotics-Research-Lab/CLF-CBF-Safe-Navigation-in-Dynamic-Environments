#!/usr/bin/env python3
"""
Moving Cylinder Follower for Gazebo.

Description
-----------
This ROS node updates a Gazebo cylinder collision model for a target
actor in the simulation. The actor pose is obtained from Gazebo model states
and the cylinder pose is updated through the Gazebo SetModelState service.

Mathematical Inputs
-------------------
- Actor pose: p_person = [x, y, z]^T

Mathematical Outputs
--------------------
- Cylinder pose: p_cylinder = [x_person, y_person, z_fixed]^T

ROS Interfaces
--------------
Subscribers:
- /gazebo/model_states (gazebo_msgs/ModelStates)

Service Clients:
- /gazebo/set_model_state (gazebo_msgs/SetModelState)

Publishers:
- None.
"""

import rospy
from gazebo_msgs.msg import ModelStates
from gazebo_msgs.msg import ModelState
from gazebo_msgs.srv import SetModelState


class ActorCollisionModel:
    """Follow a Gazebo actor by repositioning a cylinder model."""

    def __init__(self):
        """
        Initialize the node, subscriber and Gazebo service client.

        Inputs:
            None.

        Outputs:
            Initializes internal variables.

        ROS:
            Subscriber:
                /gazebo/model_states (gazebo_msgs/ModelStates)

            Service Client:
                /gazebo/set_model_state (gazebo_msgs/SetModelState)
        """

        rospy.init_node("move_cylinder_follow_person")

        rospy.wait_for_service('/gazebo/set_model_state')
        self.set_state = rospy.ServiceProxy('/gazebo/set_model_state', SetModelState)

        rospy.Subscriber('/gazebo/model_states', ModelStates, self.callback)

        self.person_pose = None

        # Cylinder height used to define its nominal vertical position.
        self.cylinder_height = 1.0
        self.z_offset = self.cylinder_height / 2.0

        self.rate = rospy.Rate(30)

    def callback(self, msg):
        """
        Retrieve and store the current pose of the target actor.

        Inputs:
            msg (gazebo_msgs/ModelStates)

        Outputs:
            Updates self.person_pose.

        ROS:
            Subscriber:
                /gazebo/model_states (gazebo_msgs/ModelStates)
        """

        try:
            idx = msg.name.index("person1")
            self.person_pose = msg.pose[idx]
        except ValueError:
            self.person_pose = None

    def run(self):
        """
        Continuously synchronize the cylinder pose with the actor position.

        Inputs:
            None.

        Outputs:
            Sends model state updates through the Gazebo service.

        ROS:
            Service Client:
                /gazebo/set_model_state (gazebo_msgs/SetModelState)
        """

        while not rospy.is_shutdown():

            if self.person_pose is not None:

                state = ModelState()
                state.model_name = "actor_collision_model"

                # Copy the actor horizontal position.
                state.pose.position.x = self.person_pose.position.x
                state.pose.position.y = self.person_pose.position.y

                # Keep the cylinder on the ground.
                state.pose.position.z = 0.0

                # Maintain a fixed upright orientation.
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
    node = ActorCollisionModel()
    node.run()
