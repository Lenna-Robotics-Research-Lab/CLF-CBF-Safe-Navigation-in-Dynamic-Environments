#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CVXPY-Based CLF/CBF Solver Node.

Description
-----------
This ROS node acts as an interface between the high-level controller and the
optimization-based controllers implemented using CVXPY. It receives a control
request containing the current robot state, reference state, and safety
information, invokes the selected optimization controller, and publishes the
resulting control command.

The controller implementation is selected at runtime using the ROS parameter
`controller_type`, with all controller-specific parameters loaded from a JSON
configuration file.

Mathematical Inputs
-------------------
Robot state:
    z = [x, y, theta]^T

Reference state:
    gamma = [gamma_x, gamma_y]^T

Safety information:
    h(z)       : Control Barrier Function (CBF) value
    ∇h(z)      : Gradient of the CBF

Mathematical Outputs
--------------------
Optimal control input:
    u = [v, omega]^T

where:
    v      : Linear velocity command [m/s]
    omega  : Angular velocity command [rad/s]

ROS Interfaces
--------------
Subscribers:
    Topic:
        clf_cbf/request
    Message:
        erl_clf_cbf_controller/ClfCbfRequest
    Description:
        Receives the robot state, reference state, and CBF information used
        to formulate the optimization problem.

Publishers:
    Topic:
        clf_cbf/result
    Message:
        erl_clf_cbf_controller/ClfCbfResult
    Description:
        Publishes the optimal velocity command together with the solver status.

Services:
    None.
"""

import rospy
import numpy as np
import cvxpy as cp

from erl_clf_cbf_controller.msg import ClfCbfRequest, ClfCbfResult
from erl_clf_cbf_controller.clf_cbf_controller import ClfCbfController
from erl_clf_cbf_controller.clf_only_controller import ClfQPController

from erl_clf_cbf_controller.clf_cbf_socp_controller_gp_map import ClfCbfSOCP_GP_MAP
from erl_clf_cbf_controller.clf_cbf_drccp_dynamic_controller import ClfCbfDrccp_dynamic_Controller

import os
import json


class CvxpySolverNode(object):
    """
    ROS interface for selecting and executing a CVXPY-based CLF/CBF controller.
    """

    def __init__(self):
        """
        Initialize the solver node, load the controller configuration, and
        create the required ROS subscriber and publisher.

        Inputs:
            None.

        Outputs:
            Initializes the selected optimization controller and ROS interfaces.

        ROS Interfaces
        --------------
        Subscriber:
            Topic:
                clf_cbf/request
            Message:
                erl_clf_cbf_controller/ClfCbfRequest

        Publisher:
            Topic:
                clf_cbf/result
            Message:
                erl_clf_cbf_controller/ClfCbfResult
        """

        rospy.init_node('clf_cbf_cvxpy_solver')

        self.controller = None

        # Controller implementation selected through a ROS parameter.
        self.controller_type = rospy.get_param(
            "~controller_type",
            "baseline_clf_cbf_qp"
        )

        self.load_controller_config()

        self.req_sub = rospy.Subscriber(
            'clf_cbf/request',
            ClfCbfRequest,
            self.req_cb,
            queue_size=10
        )

        self.res_pub = rospy.Publisher(
            'clf_cbf/result',
            ClfCbfResult,
            queue_size=10
        )

    def load_controller_config(self):
        """
        Load the controller configuration from the JSON file and initialize the
        selected controller.

        Inputs:
            None.

        Outputs:
            Loads controller parameters into member variables and calls
            init_core().

        ROS Interfaces
        --------------
        None.
        """

        current_dir = os.path.dirname(os.path.abspath(__file__))
        config_path = os.path.join(current_dir, "controller_config.json")

        with open(config_path, "r") as file:
            configs = json.load(file)

        controller_type = self.controller_type

        if controller_type not in configs:
            raise ValueError(
                "Unknown controller configuration: {}".format(controller_type)
            )

        self.config = configs[controller_type]

        params = self.config["parameters"]

        # Controller parameters loaded from the configuration file.
        self._wheel_offset = params.get("wheel_offset", 0.08)
        self._cbf_rate = params.get("cbf_rate", 0.4)
        self.noise_level = params.get("noise_level", 0.01)

        self.init_core()

    def init_core(self):
        """
        Instantiate the controller specified by the selected configuration.

        Inputs:
            None.

        Outputs:
            Creates the corresponding controller object and stores it in
            self.controller.

        ROS Interfaces
        --------------
        None.
        """

        controller_type = self.controller_type
        params = self.config["parameters"]

        # if controller_type not in [
        #     "baseline_clf_cbf_qp",
        #     "clf_qp_only",
        #     "robust_cbf_socp",
        #     "gp_cbf_socp",
        #     "drccp"
        # ]:
        #     raise ValueError("Unknown controller configuration: %s" % controller_type)

        if controller_type == "baseline_clf_cbf_qp":
            self.controller = ClfCbfController(**params)

        elif controller_type == "clf_qp_only":
            self.controller = ClfQPController(**params)

        elif controller_type == "robust_cbf_socp":
            self.controller = ClfCbf_Robust_SOCP_Controller(**params)

        elif controller_type == "gp_cbf_socp":
            self.controller = ClfCbfSOCP_GP_MAP(**params)

        elif controller_type == "drccp":
            self.controller = ClfCbfDrccp_dynamic_Controller(**params)

        else:
            raise ValueError("Unknown controller type specified.")

    def req_cb(self, msg):
        """
        Process an optimization request, compute the optimal control input,
        and publish the result.

        Inputs:
            msg (erl_clf_cbf_controller/ClfCbfRequest):
                Current robot state, desired reference, and CBF information.

        Outputs:
            Publishes a
            erl_clf_cbf_controller/ClfCbfResult
            containing the optimal linear velocity, angular velocity, and
            solver status.

        ROS Interfaces
        --------------
        Subscriber:
            Topic:
                clf_cbf/request
            Message:
                erl_clf_cbf_controller/ClfCbfRequest

        Publisher:
            Topic:
                clf_cbf/result
            Message:
                erl_clf_cbf_controller/ClfCbfResult
        """

        # Convert the incoming ROS message into optimization variables.
        rbt_pose = np.array([msg.x, msg.y, msg.theta])
        gamma_s = np.array([msg.gamma_x, msg.gamma_y])
        cbf_h_val = msg.h
        cbf_h_grad = np.array([msg.h_grad_x, msg.h_grad_y])

        res = ClfCbfResult()

        try:
            u = self.controller.generate_controller(
                rbt_pose,
                gamma_s,
                cbf_h_val,
                cbf_h_grad
            )

            res.v = float(u[0])
            res.omega = float(u[1])
            res.status = 0

        except Exception as e:
            rospy.logwarn("cvxpy solver exception: %s", str(e))

            res.v = 0.0
            res.omega = 0.0
            res.status = 1

        self.res_pub.publish(res)


if __name__ == '__main__':
    node = CvxpySolverNode()
    rospy.spin()