#!/usr/bin/python

"""
CLF-CBF Controller Post-Processing Interface.

Description
-----------
This module implements the post-processing layer of a CLF-CBF controller.
It converts the optimization output (desired linear and angular velocity)
into hardware-compatible velocity commands, applies optional actuator limits,
and publishes debugging information such as SDF values and gradient directions.

The controller assumes a unicycle-like mobile robot model where the control
input is:

    u = [v, omega]^T

where:
    v     : Linear velocity command.
    omega : Angular velocity command.

Mathematical Inputs
-------------------
Control command:
    v_dsr     : Desired linear velocity.
    w_dsr     : Desired angular velocity.

Robot pose:
    rbt_pose = [x, y, theta]

SDF gradient:
    gradient = [∂h/∂x, ∂h/∂y]

Control limits (optional):
    v_min <= v <= v_max
    w_min <= omega <= w_max

Mathematical Outputs
--------------------
Published velocity command:

    Twist.linear.x  = v
    Twist.angular.z = omega

Debug orientation:

    yaw_gradient = atan2(∂h/∂y, ∂h/∂x)

which is represented as a quaternion in a PoseStamped message.

ROS Interfaces
--------------
Publishers:
    /lmr1/cmd_vel
        Type:
            geometry_msgs/Twist
        Description:
            Publishes body-frame velocity commands.

    /sdf_val
        Type:
            std_msgs/Float64
        Description:
            Publishes signed distance field values.

    /clf_cbf_debug
        Type:
            geometry_msgs/PoseStamped
        Description:
            Publishes debugging visualization containing robot position and
            SDF gradient direction.

Subscribers:
    None.

Services:
    None.
"""

import rospy
import numpy as np

from geometry_msgs.msg import Twist, PoseStamped
from std_msgs.msg import Float64, Header
from tf.transformations import quaternion_from_euler


def clip(x, x_min, x_max):
    """
    Saturate a scalar value within predefined bounds.

    Inputs:
        x (float):
            Input value.

        x_min (float):
            Minimum allowed value.

        x_max (float):
            Maximum allowed value.

    Outputs:
        float:
            Clipped value satisfying:
                x_min <= x <= x_max.

    ROS:
        None.
    """

    if x < x_min:
        x = x_min

    if x > x_max:
        x = x_max

    return x


class ClfCbfPostprocess:
    """
    Post-processing interface between CLF-CBF controller outputs and robot
    velocity commands.
    """

    def __init__(self, ctrl_limits=None):
        """
        Initialize ROS publishers and optional control saturation limits.

        Inputs:
            ctrl_limits (dict, optional):
                Velocity constraints:
                    v_min, v_max,
                    w_min, w_max.

        Outputs:
            Initializes publishers and internal ROS messages.

        ROS Interfaces
        --------------
        Publishers:
            /lmr1/cmd_vel (geometry_msgs/Twist)
            /sdf_val (std_msgs/Float64)
            /clf_cbf_debug (geometry_msgs/PoseStamped)
        """

        # Velocity command publisher for the mobile robot.
        self.cmd_vel_pub = rospy.Publisher(
            '/lmr1/cmd_vel',
            Twist,
            queue_size=1
        )

        # SDF value visualization publisher.
        self.sdf_val_pub = rospy.Publisher(
            '/sdf_val',
            Float64,
            queue_size=1
        )

        # Debug visualization publisher.
        self.debug_msg_pub = rospy.Publisher(
            '/clf_cbf_debug',
            PoseStamped,
            queue_size=1
        )

        self._body_twist = Twist()
        self._sdf_val = Float64()

        self._sdf_grad = PoseStamped()
        self._sdf_grad.header = Header()
        self._sdf_grad.header.frame_id = 'map'

        if ctrl_limits is not None:
            self.ctrl_limits = ctrl_limits

        rospy.loginfo("[unicycle controller post-processor initialized!]")

    def send_cmd(self, v_dsr, w_dsr, clip_ctrl=False, debug=False):
        """
        Publish desired linear and angular velocity commands.

        Inputs:
            v_dsr (float):
                Desired linear velocity.

            w_dsr (float):
                Desired angular velocity.

            clip_ctrl (bool):
                Enables velocity saturation if True.

            debug (bool):
                Enables throttled debug logging.

        Outputs:
            Publishes geometry_msgs/Twist command.

        ROS Interfaces
        --------------
        Publisher:
            /lmr1/cmd_vel (geometry_msgs/Twist)
        """

        if debug:
            rospy.logwarn_throttle(
                0.5,
                "Input body twist (v_dsr, omega_dsr) [%.2f, %.2f]"
                % (v_dsr, w_dsr)
            )

        if clip_ctrl and self.ctrl_limits is not None:
            v_dsr = clip(
                v_dsr,
                self.ctrl_limits['v_min'],
                self.ctrl_limits['v_max']
            )

            w_dsr = clip(
                w_dsr,
                self.ctrl_limits['w_min'],
                self.ctrl_limits['w_max']
            )

        if debug:
            rospy.logwarn_throttle(
                0.5,
                "Output body twist (v_dsr, omega_dsr) [%.2f, %.2f]"
                % (v_dsr, w_dsr)
            )

        self._body_twist.linear.x = v_dsr
        self._body_twist.angular.z = w_dsr

        self.cmd_vel_pub.publish(self._body_twist)

    def send_sdf_val(self, sdf_server_result):
        """
        Publish the current signed distance field value.

        Inputs:
            sdf_server_result (std_msgs/Float64):
                SDF value received from the SDF computation module.

        Outputs:
            Publishes the SDF value.

        ROS Interfaces
        --------------
        Publisher:
            /sdf_val (std_msgs/Float64)
        """

        self.sdf_val_pub.publish(sdf_server_result)

    def send_debug(self, rbt_pose, gradient):
        """
        Publish robot pose and SDF gradient direction for visualization.

        Inputs:
            rbt_pose (array-like):
                Robot pose:
                    [x, y, theta]

            gradient (array-like):
                SDF gradient:
                    [∂h/∂x, ∂h/∂y]

        Outputs:
            Publishes a PoseStamped message representing the gradient direction.

        ROS Interfaces
        --------------
        Publisher:
            /clf_cbf_debug (geometry_msgs/PoseStamped)
        """

        # Update visualization timestamp.
        self._sdf_grad.header.stamp = rospy.Time.now()

        # Publish robot position.
        self._sdf_grad.pose.position.x = rbt_pose[0]
        self._sdf_grad.pose.position.y = rbt_pose[1]
        self._sdf_grad.pose.position.z = 0

        # Convert gradient direction into quaternion orientation.
        quaternion = quaternion_from_euler(
            0.0,
            0.0,
            np.arctan2(gradient[1], gradient[0])
        )

        self._sdf_grad.pose.orientation.x = quaternion[0]
        self._sdf_grad.pose.orientation.y = quaternion[1]
        self._sdf_grad.pose.orientation.z = quaternion[2]
        self._sdf_grad.pose.orientation.w = quaternion[3]

        self.debug_msg_pub.publish(self._sdf_grad)