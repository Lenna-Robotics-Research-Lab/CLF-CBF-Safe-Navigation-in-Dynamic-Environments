#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CLF-CBF-QP Controller for Unicycle Mobile Robots.

Description
-----------
This module implements a Control Lyapunov Function (CLF) and Control Barrier
Function (CBF) based Quadratic Program (QP) controller for a unicycle-like
mobile robot. The controller generates safe linear and angular velocity
commands while simultaneously driving the robot toward a desired goal.

Mathematical Inputs
-------------------
rbt_pose : np.ndarray
    Current robot pose [x, y, theta].

gamma_s : np.ndarray
    Desired reference position [x_d, y_d].

cbf_h_val : float
    Control Barrier Function value representing obstacle clearance or safety
    margin.

cbf_h_grad : np.ndarray
    Gradient of the Control Barrier Function with respect to the robot state.

Mathematical Outputs
--------------------
u = [v, omega]
    v     : Linear velocity command [m/s].
    omega : Angular velocity command [rad/s].

Optimization Problem
--------------------
The controller solves a CLF-CBF Quadratic Program of the form:

    min J(u)

subject to:
    CLF constraint  -> goal convergence
    CBF constraint  -> obstacle avoidance
    Input bounds    -> actuator limits

No ROS publishers, subscribers, services, or actions are used in this module.
This file is a standalone optimization-based controller implementation.
"""

import numpy as np
import cvxpy as cp


class ClfCbfController:
    """
    CLF-CBF-QP controller for generating safe velocity commands for a
    unicycle robot.
    """

    def __init__(self, p1=1e0, p2=1e0, p3=1e2, clf_rate=1.0,
                 cbf_rate=0.25, wheel_offset=0.08, noise_level=0.01):
        """
        Initialize controller parameters, optimization weights, and internal state.

        Inputs:
            p1 (float): Linear velocity objective weight.
            p2 (float): Angular velocity objective weight.
            p3 (float): CLF slack variable penalty.
            clf_rate (float): CLF decay rate.
            cbf_rate (float): CBF decay rate.
            wheel_offset (float): Off-wheel control point distance.
            noise_level (float): Reserved parameter for future noise modeling.

        Outputs:
            None.

        ROS:
            No publishers, subscribers, or services.
        """

        # Optimization weights
        self.p1 = p1
        self.p2 = p2
        self.p3 = p3

        # CLF and CBF decay rates
        self.rateV = clf_rate
        self.rateh = cbf_rate

        # CLF shaping gains
        self.linear_gain_sq = 0.05
        self.angular_gain_sq = 0.4

        # Off-wheel control point distance
        self.l = wheel_offset

        # Solver status flag
        self.solve_fail = True

        # Velocity constraints and previous control input
        self.max_v = 1.2
        self.prev_u = np.zeros(2)

        print("Off-wheel: ", self.l)

    def generate_controller(self, rbt_pose, gamma_s, cbf_h_val, cbf_h_grad):
        """
        Generate linear and angular velocity commands using a CLF-CBF-QP formulation.

        Inputs:
            rbt_pose (np.ndarray): Current robot pose [x, y, theta].
            gamma_s (np.ndarray): Desired reference position.
            cbf_h_val (float): CBF value.
            cbf_h_grad (np.ndarray): CBF gradient.

        Outputs:
            np.ndarray:
                [v, omega] optimal control input.

        ROS:
            No publishers, subscribers, or services.
        """
        # Projection vectors defining robot heading and lateral directions
        proj_perp = -np.array([[-np.sin(rbt_pose[2])],
                               [np.cos(rbt_pose[2])]])

        proj_alog = -np.array([[np.cos(rbt_pose[2])],
                               [np.sin(rbt_pose[2])]])

        # CLF state transformation
        m_z = proj_perp.T @ (rbt_pose[0:2] - gamma_s[0:2])
        n_z = proj_alog.T @ (rbt_pose[0:2] - gamma_s[0:2])

        # CLF candidate function
        V = 0.5 * (
            self.linear_gain_sq *
            ((rbt_pose[0] - gamma_s[0]) ** 2 +
             (rbt_pose[1] - gamma_s[1]) ** 2)
            +
            self.angular_gain_sq *
            np.arctan2(m_z, n_z) ** 2
        )

        dV_dxy = (
            self.linear_gain_sq * (rbt_pose[0:2] - gamma_s[0:2]).T
            +
            self.angular_gain_sq *
            np.arctan2(m_z, n_z) *
            (1 / (m_z ** 2 + n_z ** 2)) *
            (n_z * proj_perp.T - m_z * proj_alog.T)
        )

        dV_dtheta = -self.angular_gain_sq * np.arctan2(m_z, n_z)

        dVdx = np.row_stack((dV_dxy[0, 0], dV_dxy[0, 1], dV_dtheta))

        # Unicycle input matrix with off-wheel control point
        g_x = np.array([
            [np.cos(rbt_pose[2]), -self.l * np.sin(rbt_pose[2])],
            [np.sin(rbt_pose[2]), self.l * np.cos(rbt_pose[2])],
            [0, 1]
        ])

        # CBF derivative mapping
        dot_h = np.array([[
            np.cos(rbt_pose[2]) * cbf_h_grad[0] +
            np.sin(rbt_pose[2]) * cbf_h_grad[1],

            -self.l * np.sin(rbt_pose[2]) * cbf_h_grad[0] +
            self.l * np.cos(rbt_pose[2]) * cbf_h_grad[1]
        ]])

        uu = cp.Variable(3)

        # CLF-CBF-QP constraints
        constraints = [
            dVdx.T @ (g_x @ uu[:2]) + self.rateV * V <= uu[2],
            dot_h[0][0] * uu[0] + dot_h[0][1] * uu[1]
            + self.rateh * (cbf_h_val - 0.2) >= 0,
            uu[2] >= 0.0,
            cp.abs(uu[0]) <= self.max_v,
            cp.abs(uu[1]) <= 1.00
        ]

        # Adaptive objective weights based on obstacle proximity
        self.p3 = 5 * cbf_h_val
        self.p1 = 2 * cbf_h_val

        self.p0 = 4
        self.p2 = 0.4

        obj = cp.Minimize(
            self.p0 * cp.norm(self.prev_u - uu[0:2]) ** 2 +
            self.p1 * cp.square(uu[0] - self.max_v) +
            self.p2 * cp.square(uu[1]) +
            self.p3 * cp.square(uu[2])
        )

        prob = cp.Problem(obj, constraints)

        # Solve CLF-CBF-QP
        prob.solve(solver='SCS', verbose=False)

        if prob.status == "infeasible":
            self.solve_fail = True

            print("-------------------------- SOLVER NOT OPTIMAL -------------------------")
            print("[In solver] solver status: ", prob.status)
            print("[In solver] h = ", cbf_h_val)
            print("[In solver] dot_h = ", dot_h)

            self.prev_u = np.array([0.0, 0.0])

            return np.array([0.0, 0.0])

        self.solve_fail = False
        self.prev_u = uu[0:2].value

        # Closed-loop CLF metrics
        self.V_val = V
        self.dotV_val = dVdx.T @ (g_x @ uu.value[0:2])

        return uu.value[0:2]
