#!/usr/bin/env python3
"""
CLF-CBF Controller Wrapper for Unicycle Mobile Robots.

Description
-----------
This node implements the low-level control interface between robot state
estimation, reference generation, obstacle information, and an external
CLF-CBF optimization backend.

Mathematical Inputs
-------------------
- Robot state z = [x, y, theta]^T from odometry/localization.
- Desired state z* (local goal/reference governor output).
- Control Barrier Function (CBF) values and gradients obtained from either:
  * Signed Distance Field (SDF) queries, or
  * LiDAR / obstacle point cloud processing.
- Optional moving obstacle states.

Mathematical Outputs
--------------------
- Linear velocity command v.
- Angular velocity command w.
- Safety metrics including minimum CBF/SDF value and gradient.

ROS Interfaces
--------------
Subscribers:
- clf_cbf/result (erl_clf_cbf_controller/ClfCbfResult)
- Additional subscribers are instantiated inside ClfCbfPreprocess.

Publishers:
- clf_cbf/request (erl_clf_cbf_controller/ClfCbfRequest)
- Additional publishers are instantiated inside ClfCbfPostprocess and
  ClfCbfPreprocess.

Services:
- Optional SDF service client used for obstacle-distance evaluation.

Control Flow
------------
1. Receive robot state and desired state.
2. Compute obstacle-distance information and CBF quantities.
3. Formulate and send a CLF-CBF optimization request.
4. Receive optimal control solution.
5. Publish safe velocity commands.
"""

from __future__ import print_function
import rospy
import numpy as np
import numpy.linalg as npla
import time
from erl_clf_cbf_controller.pre_process import ClfCbfPreprocess
from erl_clf_cbf_controller.post_process import ClfCbfPostprocess

from erl_clf_cbf_controller.msg import ClfCbfRequest, ClfCbfResult

import json

# different controller formulation
# from erl_clf_cbf_controller.clf_cbf_controller import ClfCbfController
# from erl_clf_cbf_controller.clf_only_controller import ClfQPController

# from erl_clf_cbf_controller.clf_cbf_socp_controller_gp_map import ClfCbfSOCP_GP_MAP
# from erl_clf_cbf_controller.clf_cbf_drccp_dynamic_controller import ClfCbfDrccp_dynamic_Controller

import csv
import os


def off_wheel_map(rbt_pose, off_wheel_length):
    """
    Map the robot pose from the center of mass to the off-wheel control point.

    Inputs:
        rbt_pose (np.ndarray): [x, y, theta].
        off_wheel_length (float): Offset distance from robot center.

    Outputs:
        np.ndarray: [x_off, y_off, theta].

    ROS:
        No publishers/subscribers.
    """
    off_wheel_x = rbt_pose[0] + off_wheel_length * np.cos(rbt_pose[2])
    off_wheel_y = rbt_pose[1] + off_wheel_length * np.sin(rbt_pose[2])
    off_wheel_theta = rbt_pose[2]

    return np.array([off_wheel_x, off_wheel_y, off_wheel_theta])


class ClfCbfControllerWrapper:
    """
    CLF-CBF-QP controller wrapper, check system status
    """

    # status table
    NORMAL = 0
    GOAL_REACHED = 1

    def __init__(self, config_table=None):
        """Init ClfCbfControllerWrapper class.
        This controller subscribes:
            odom (from localization)
            desired robot states (from high level controller, i.e., ref_)
        Publish:
            desired velocity / body twist
        """

        # function class handle
        self.preprocessor = None
        self.core = None
        self.postprocessor = None

        # loading external config parameters
        self._config_dict = config_table

        self.controller_type = rospy.get_param("~controller_type", "baseline_clf_cbf_qp")
        self.load_controller_config()

        # system status
        self.status = ClfCbfControllerWrapper.NORMAL

        # -------------------- Constants --------------------
        self.ctrl_limits = self._config_dict["ctrl_limits"]
        rospy.logwarn("self.ctrl_limits %s" % self.ctrl_limits)

        # -------------------- Init Modules --------------------
        self.init_preprocessor()
        self.init_postprocessor()

        # -------------------- Cache --------------------
        self.response_cache = []

        # -------------------- for plotting performance --------------------

        self.v_list = []
        self.w_list = []
        self.distance_list = []

        self.error_to_gamma = []

        self.robot_pos = []

        # -------------------- for plotting moving obstacles in dynamic env --------------------

        self.robot_started_moving = False
        self.mov_obs_flag = False
        self.mov_obs_data = []
        self.waypoints = []
        self.last_waypoint = None

        # set numpy array console print precision = 2
        np.set_printoptions(formatter={"float": "{: 0.2f}".format})
        rospy.loginfo("CLF-CBF CONTROL NODE INIT SUCCESSFUL!")

        self.qp_req_pub = rospy.Publisher('clf_cbf/request',
                                          ClfCbfRequest, queue_size=1)
        self.qp_res_sub = rospy.Subscriber('clf_cbf/result',
                                           ClfCbfResult,
                                           self.qp_result_cb,
                                           queue_size=1)
        self.latest_result = None

    def qp_result_cb(self, msg):
        """
        Store the latest optimization result received from clf_cbf/result.

        Input:
            msg (ClfCbfResult)

        Output:
            Updates self.latest_result.

        ROS:
            Subscriber: clf_cbf/result (ClfCbfResult)
        """
        self.latest_result = msg

    def send_qp_request_and_get_result(self, req, timeout=1, rate_hz=100):
        """
        Publish a ClfCbfRequest and wait up to 'timeout' seconds
        for a ClfCbfResult. Returns (v, w, ok).
        """
        self.latest_result = None
        self.qp_req_pub.publish(req)

        rate = rospy.Rate(rate_hz)
        steps = int(timeout * rate_hz)

        for _ in range(steps):
            if self.latest_result is not None:
                if self.latest_result.status == 0:
                    return self.latest_result.v, self.latest_result.omega, True
                else:
                    return 0.0, 0.0, False
            rate.sleep()

        return 0.0, 0.0, False


    def load_controller_config(self):
        # Load the configuration file
        current_dir = os.path.dirname(os.path.abspath(__file__))
        config_path = os.path.join(current_dir, "controller_config.json")
        with open(config_path, "r") as file:
            configs = json.load(file)

        # Determine which controller configuration to use
        controller_type = self.controller_type

        if controller_type not in configs:
            raise ValueError("Unknown controller configuration: {}".format(controller_type))

        self.config = configs[controller_type]

        # Assigning new variables from the selected config
        params = self.config["parameters"]
        self._wheel_offset = params.get("wheel_offset", 0.1)    # Default to 0.08 if not set
        self._cbf_rate = params.get("cbf_rate", 0.4)            # Default to 0.4 if not set
        self.noise_level = params.get("noise_level", 0.01)      # Default to 0.01 if not set

        # Initialize the controller based on the controller type
        # self.init_core()

    def init_preprocessor(self):
        """
        Init preprocessor of node.
        """
        self.preprocessor = ClfCbfPreprocess()

    def save_rob_data_to_csv(self, filename):
        """Save collected data to a CSV file."""
        # Ensure directory exists
        script_dir = os.path.dirname(os.path.realpath(__file__))
        file_path = os.path.join(script_dir, filename)

        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(
                [
                    "Linear Velocity",
                    "Angular Velocity",
                    "Distance to Obstacle",
                    "Tracking Error",
                    "X Position",
                    "Y Position",
                ]
            )

            for v, w, dist, tracking_error, pos in zip(
                self.v_list, self.w_list, self.distance_list, self.error_to_gamma, self.robot_pos
            ):
                writer.writerow([v, w, dist, tracking_error, pos[0][0], pos[1][0]])

    def save_mov_obs_data_to_csv(self, filename="mov_obs_data.csv"):
        """Save collected moving obstacles data to a CSV file."""
        script_dir = os.path.dirname(os.path.realpath(__file__))
        file_path = os.path.join(script_dir, filename)

        with open(file_path, "w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(["PosX", "PosY", "VelX", "VelY", "Radius"])
            for data in self.mov_obs_data:
                writer.writerow(data)

    def save_waypoints_to_csv(self, filename="waypoints.csv"):
        """Save collected waypoints to a CSV file."""
        script_dir = os.path.dirname(os.path.realpath(__file__))
        file_path = os.path.join(script_dir, filename)

        with open(file_path, "w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(["X", "Y", "Theta"])
            for waypoint in self.waypoints:
                writer.writerow(waypoint)

    def init_core(self):
        """
        Init unicycle controller. For each controller the interface might be different.
        More details in controller itself.
        """
        # Initialize the controller based on the configuration
        controller_type = self.controller_type
        # params = self.config["parameters"]

        if controller_type not in [
            "baseline_clf_cbf_qp",
            "clf_qp_only",
            "robust_cbf_socp",
            "gp_cbf_socp",
            "drccp"
        ]:
            raise ValueError("Unknown controller configuration: %s" % controller_type)

        # if controller_type == "baseline_clf_cbf_qp":
        #     self.core = ClfCbfController(**params)
        # elif controller_type == "clf_qp_only":
        #     self.core = ClfQPController(**params)
        # elif controller_type == "robust_cbf_socp":
        #     self.core = ClfCbf_Robust_SOCP_Controller(**params)
        # elif controller_type == "gp_cbf_socp":
        #     self.core = ClfCbfSOCP_GP_MAP(**params)
        # elif controller_type == "drccp":
        #     self.core = ClfCbfDrccp_dynamic_Controller(**params)
        # else:
        #     raise ValueError("Unknown controller type specified.")

    def init_postprocessor(self):
        """
        Init preprocessor of node.
        """
        self.postprocessor = ClfCbfPostprocess(ctrl_limits=self.ctrl_limits)

    def update_waypoints(self):
        current_waypoint = self.preprocessor._np_z_dsr.tolist()
        if self.last_waypoint != current_waypoint:  # Check if the waypoint has changed
            self.waypoints.append(current_waypoint)
            self.last_waypoint = current_waypoint  # Update the last waypoint

    def update(self, eps_dist=0.25, eps_dist_reset=0.35, num_samples=5):
        """
        Update loop as follows:
            1. collect latest data from preprocessor (callback automatically)
            2. execuate update loop using core
            3. sending command to downstream via post-processor
        """

        # update the waypoint (currently, it is just local goal)
        self.update_waypoints()

        # receive robot pose from process
        z = off_wheel_map(rbt_pose=self.preprocessor._np_z, off_wheel_length=0.15)
        rbt_xy = np.array([[self.preprocessor._np_z[0]], [self.preprocessor._np_z[1]]])

        # obtain local goal on path using simplified reference governor
        z_dsr = off_wheel_map(rbt_pose=self.preprocessor._np_z_dsr, off_wheel_length=0.15)

        # obtain distance error
        dist_err = npla.norm(self.preprocessor._np_z[0:2] - z_dsr[0:2])

        # start with default zero control
        v, w = 0.0, 0.0

        # obtain cbf value and gradient from sdf server
        if self.preprocessor.use_sdf:
            start = time.time()
            while True:
                try:
                    response_temp = self.preprocessor.sdf_client([z[0]], [z[1]])
                    response_cache = self.response_cache
                    break
                except rospy.ServiceException as e:
                    rospy.logwarn_throttle(0.5, "Service call failed: {}. Retrying...".format(e))
            end = time.time()
            print("ros sdf_client: {} sec".format(end - start))

            if np.isnan(np.array([response_temp.sdf])):
                response = self.response_cache
            else:
                response = response_temp
                self.response_cache = response_temp

            msg_sdf = self.preprocessor.msg_sdf
            msg_sdf.header.stamp = rospy.Time.now()
            msg_sdf.header.seq += 1
            msg_sdf.pose.pose.position.x = z[0]
            msg_sdf.pose.pose.position.y = z[1]
            msg_sdf.pose.covariance[0] = response.sdf[0] ** 2
            msg_sdf.pose.covariance[7] = response.sdf[0] ** 2
            rospy.loginfo_throttle(0.5, "Received SDF response: {}".format(response.sdf[0]))
            self.preprocessor.sdf_pub.publish(msg_sdf)

            cbf_value = np.array([response.sdf])
            cbf_grad = np.array([response.gradient_x, response.gradient_y])

            # print('gp_cbf:', cbf_value)
            # print('gp_cbf_grad:', cbf_grad)

        else:
            cbf_value = np.zeros((num_samples,))
            cbf_grad = np.zeros((2, num_samples))
            partial_h_partial_t_static = np.zeros((num_samples,))

            for index, item in enumerate(self.preprocessor.scan_buffer):
                dist_rbt2surface = np.linalg.norm(rbt_xy - item, axis=0)

                """
                Adding lidar noise (naive way), it is highly recommended to modClfCbfControllerWrapperify the source urdf file instead
                """
                # Adding Gaussian noise to each distance measurement

                # sigma_values = self.noise_level * dist_rbt2surface

                # noise = np.random.normal(0, sigma_values)
                # dist_rbt2surface = dist_rbt2surface + noise

                index_closest = np.argmin(dist_rbt2surface)
                if np.size(index_closest) > 1:
                    index_closest = index_closest[0]

                cbf_value[index] = dist_rbt2surface[index_closest]

                # Calculate the gradient
                gradient = rbt_xy[:, 0] - item[:, index_closest]

                # Normalize the gradient to have a 2-norm of 1
                norm_gradient = gradient / np.linalg.norm(gradient)
                cbf_grad[:, index] = norm_gradient

            # static obstacle cbf has no time change
            partial_h_partial_t = partial_h_partial_t_static

            # calculate cbf related parameters for moving obstacles
            if self.preprocessor.mov_obs:
                if self.preprocessor.mov_obs_ready:
                    self.mov_obs_flag = True

                    mov_obs_xy = np.vstack((self.preprocessor.mo_list[:, 0].T, self.preprocessor.mo_list[:, 1].T))
                    mov_obs_rad = self.preprocessor.mo_list[:, 4]
                    mov_obs_vel = np.vstack((self.preprocessor.mo_list[:, 2].T, self.preprocessor.mo_list[:, 3].T))

                    # if self.robot_started_moving:
                    # # saving data for visualization
                    #     for i in range(len(mov_obs_xy[0])):
                    #         self.mov_obs_data.append([mov_obs_xy[0][i], mov_obs_xy[1][i], mov_obs_vel[0][i], mov_obs_vel[1][i], mov_obs_rad[i]])

                    dist_rbt2mov = np.linalg.norm(rbt_xy - mov_obs_xy, axis=0) - mov_obs_rad
                    mov_obs_grad = rbt_xy - mov_obs_xy
                    norm_mov_gradient = mov_obs_grad / np.linalg.norm(mov_obs_grad, axis=0)

                    cbf_grad = np.concatenate((norm_mov_gradient, cbf_grad), axis=1)
                    cbf_value = np.concatenate((dist_rbt2mov, cbf_value))
                    
                    # calculate the time change of cbf caused by moving obstacles
                    #here it may cause a problem and to fix it we can run the code below 
                    # partial_h_partial_t_mov = -1.0 * (norm_mov_gradient * mov_obs_vel).sum(axis=1)

                    partial_h_partial_t_mov = (-1.0 * np.dot(norm_mov_gradient.T, mov_obs_vel)).diagonal()
                    partial_h_partial_t = np.concatenate((partial_h_partial_t_mov, partial_h_partial_t))

            # obtain smallest five samples
            cbf_index = np.argsort((cbf_value - 0.15) * self._cbf_rate + partial_h_partial_t)[0:5]
            sorted_cbf_value = cbf_value[cbf_index]
            sorted_cbf_grad = cbf_grad[:, cbf_index]
            sorted_partial_h_partial_t = partial_h_partial_t[cbf_index]

            # print('sorted_dh_dt:', partial_h_partial_t_mov)

        min_index = np.argmin(cbf_value)  # Find the index of the minimum cbf_value

        # Set sdf_val as the min of cbf_value
        sdf_val = cbf_value[min_index]

        # Set sdf_grad as the corresponding cbf_grad
        sdf_grad = cbf_grad[:, min_index]

        # send as ros msg to measure time
        self.postprocessor.send_sdf_val(sdf_val)

        # start with pose_reached status
        if self.status == ClfCbfControllerWrapper.GOAL_REACHED:
            if dist_err > eps_dist_reset:
                self.status = ClfCbfControllerWrapper.NORMAL
                msg = "[clf-cbf controller] status [down] [goal --> normal] triggered by [dist] err"
                msg += ": dist err > eps_dist_reset (%.3f > %.3f)" % (dist_err, eps_dist_reset)
                rospy.logwarn_throttle(0.5, msg)

        if self.status == ClfCbfControllerWrapper.NORMAL:
            if dist_err <= eps_dist:
                self.status = ClfCbfControllerWrapper.GOAL_REACHED
                msg = "[clf-cbf controller] status [ up ] [normal --> goal]"
                msg += ": dist err <= eps_dist (%.3f <= %.3f)" % (dist_err, eps_dist)
                rospy.logwarn_throttle(0.5, msg)

        # run status table
        if self.status == ClfCbfControllerWrapper.GOAL_REACHED:
            v, w = 0.0, 0.0
        elif self.status == ClfCbfControllerWrapper.NORMAL:
            # generate control value
            if self.controller_type == "baseline_clf_cbf_qp":
                req = ClfCbfRequest()
                req.x = z[0]
                req.y = z[1]
                req.theta = z[2]
                req.gamma_x = z_dsr[0]
                req.gamma_y = z_dsr[1]
                req.h = sdf_val
                req.h_grad_x = sdf_grad[0]
                req.h_grad_y = sdf_grad[1]

                v, w, ok = self.send_qp_request_and_get_result(req)
                if not ok:
                    v, w = 0.0, 0.0

            elif self.controller_type == "clf_qp_only":
                req = ClfCbfRequest()
                req.x = z[0]
                req.y = z[1]
                req.theta = z[2]
                req.gamma_x = z_dsr[0]
                req.gamma_y = z_dsr[1]
                req.h = sdf_val
                req.h_grad_x = sdf_grad[0]
                req.h_grad_y = sdf_grad[1]

                v, w, ok = self.send_qp_request_and_get_result(req)
                if not ok:
                    v, w = 0.0, 0.0

            elif self.controller_type == "robust_cbf_socp":
                # For now, send only the basic fields; robust variants would
                # require an extended message type if you need the extra arrays.
                req = ClfCbfRequest()
                req.x = z[0]
                req.y = z[1]
                req.theta = z[2]
                req.gamma_x = z_dsr[0]
                req.gamma_y = z_dsr[1]
                req.h = sdf_val
                req.h_grad_x = sdf_grad[0]
                req.h_grad_y = sdf_grad[1]
                # h_error_bound and grad_h_error_bound would need extra msg fields

                v, w, ok = self.send_qp_request_and_get_result(req)
                if not ok:
                    v, w = 0.0, 0.0

            elif self.controller_type == "gp_cbf_socp":
                req = ClfCbfRequest()
                req.x = self.preprocessor._np_z[0]
                req.y = self.preprocessor._np_z[1]
                req.theta = self.preprocessor._np_z[2]
                req.gamma_x = z_dsr[0]
                req.gamma_y = z_dsr[1]
                req.h = sdf_val
                req.h_grad_x = sdf_grad[0]
                req.h_grad_y = sdf_grad[1]
                # cbf_h_variance and cbf_h_grad_var would also need extra msg fields

                v, w, ok = self.send_qp_request_and_get_result(req)
                if not ok:
                    v, w = 0.0, 0.0

            elif self.controller_type == "drccp":
                # DRCCP needs more data than the current ClfCbfRequest carries;
                # you would need a richer message if you really want the full DRCCP logic.
                req = ClfCbfRequest()
                req.x = z[0]
                req.y = z[1]
                req.theta = z[2]
                req.gamma_x = z_dsr[0]
                req.gamma_y = z_dsr[1]
                req.h = sdf_val
                req.h_grad_x = sdf_grad[0]
                req.h_grad_y = sdf_grad[1]
                # h_samples, h_grad_samples, dh_dt_samples are not transmitted yet

                v, w, ok = self.send_qp_request_and_get_result(req)
                if not ok:
                    v, w = 0.0, 0.0
            else:
                raise ValueError("Unknown controller type: {}".format(self.controller_type))

            # for plotting purposes

            # Check if the robot has started moving
            if abs(v) > 0.001 or abs(w) > 0.001:
                self.robot_started_moving = True

            if self.mov_obs_flag:
                for i in range(len(mov_obs_xy[0])):
                    self.mov_obs_data.append(
                        [mov_obs_xy[0][i], mov_obs_xy[1][i], mov_obs_vel[0][i], mov_obs_vel[1][i], mov_obs_rad[i]]
                    )

            self.v_list.append(v)
            self.w_list.append(w)
            if self.preprocessor.use_sdf:
                self.distance_list.append(sdf_val[0] - 0.15)
            else:
                self.distance_list.append(np.min(sorted_cbf_value) - 0.15)
            self.error_to_gamma.append(dist_err)
            self.robot_pos.append(rbt_xy)

        else:
            # reserve for more status
            pass

        self.postprocessor.send_cmd(v_dsr=v, w_dsr=w, clip_ctrl=True)
        self.postprocessor.send_debug(rbt_pose=z, gradient=sdf_grad)


if __name__ == "__main__":
    try:
        rospy.init_node("clf_cbf_controller")
        rospy.loginfo("[clf_cbf_controller] Started!\n")

        # loading parameters
        ctrl_freq = rospy.get_param("~ctrl_freq", 50.0)

        # control limit
        v_min = rospy.get_param("~v_min", -2.0)
        v_max = rospy.get_param("~v_max", 2.0)

        w_min = rospy.get_param("~w_min", 2.0)
        w_max = rospy.get_param("~w_max", -2.0)

        config_dict = {"ctrl_limits": {"v_min": v_min, "v_max": v_max, "w_min": w_min, "w_max": w_max}}

        clf_cbf_controller = ClfCbfControllerWrapper(config_dict)
        rate = rospy.Rate(ctrl_freq)

        while not rospy.is_shutdown():
            if clf_cbf_controller.preprocessor._upstream_init_finish:
                clf_cbf_controller.update()
            else:
                pass
            rate.sleep()

    except rospy.ROSInterruptException:
        # clf_cbf_controller.save_rob_data_to_csv('saved_data/path2/test.csv')
        # clf_cbf_controller.save_mov_obs_data_to_csv('saved_data/dynamic/mov_obs_data.csv')
        # clf_cbf_controller.save_waypoints_to_csv('saved_data/dynamic/waypoints.csv')
        pass