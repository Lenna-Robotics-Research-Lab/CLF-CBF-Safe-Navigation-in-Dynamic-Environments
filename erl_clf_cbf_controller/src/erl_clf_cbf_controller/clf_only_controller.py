#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CLF-QP Controller for Unicycle Mobile Robots.

Description
-----------
This module implements a Control Lyapunov Function (CLF) based Quadratic
Program (QP) controller for a unicycle-like mobile robot. Unlike the
CLF-CBF-QP controller, this implementation does not enforce Control Barrier
Function (CBF) constraints in the optimization problem. The controller
generates velocity commands that drive the robot toward a desired reference
while satisfying actuator limits.

Mathematical Inputs
-------------------
rbt_pose : np.ndarray
    Current robot pose:
        z = [x, y, theta]^T

gamma_s : np.ndarray
    Desired reference position:
        gamma = [x_d, y_d]^T

cbf_h_val : float
    Safety metric or obstacle distance value.
    In this implementation, it is used only for adaptive objective weighting
    and does not appear in the optimization constraints.

cbf_h_grad : np.ndarray
    Gradient of the CBF. Currently unused in the optimization formulation.

Mathematical Outputs
--------------------
u = [v, omega]

where:
    v     : Linear velocity command [m/s]
    omega : Angular velocity command [rad/s]

Optimization Problem
--------------------
The controller solves a Quadratic Program of the form:

    minimize J(v, omega, delta)

subject to:
    CLF constraint
    Input saturation constraints

where the CLF constraint guarantees convergence toward the reference point.
"""

import numpy as np
import cvxpy as cp


class ClfQPController:
    """
    CLF-based Quadratic Program controller for a unicycle mobile robot.
    """

    def __init__(self, p1=1e0, p2=1e0, p3=1e2,
                 clf_rate=1.0, cbf_rate=0.25,
                 wheel_offset=0.08, noise_level=0.01):
        """
        Initialize controller parameters and optimization settings.

        Inputs:
            p1 (float):
                Linear velocity objective weight.

            p2 (float):
                Angular velocity objective weight.

            p3 (float):
                CLF slack variable penalty weight.

            clf_rate (float):
                Control Lyapunov Function decay rate.

            cbf_rate (float):
                Reserved parameter retained for interface consistency.

            wheel_offset (float):
                Distance between the robot center and off-wheel control point.

            noise_level (float):
                Reserved parameter for future noise modeling.

        Outputs:
            None.

        ROS:
            Publishers : None
            Subscribers: None
            Services   : None
        """

        # Optimization weights
        self.p1 = p1
        self.p2 = p2
        self.p3 = p3

        # CLF and CBF rates
        self.rateV = clf_rate
        self.rateh = cbf_rate

        # CLF shaping gains
        self.linear_gain_sq = 0.05
        self.angular_gain_sq = 0.4

        # Off-wheel control point distance
        self.l = wheel_offset

        # Solver status
        self.solve_fail = True

        # Velocity limits and previous control input
        self.max_v = 1.2
        self.prev_u = np.zeros(2)

        print("Off-wheel: ", self.l)

    def generate_controller(self, rbt_pose, gamma_s,
                            cbf_h_val, cbf_h_grad):
        """
        Generate optimal control inputs using a CLF-QP formulation.

        Inputs:
            rbt_pose (np.ndarray):
                Current robot pose [x, y, theta].

            gamma_s (np.ndarray):
                Desired reference position.

            cbf_h_val (float):
                Safety metric used only for adaptive objective weighting.

            cbf_h_grad (np.ndarray):
                Unused argument retained for compatibility with other
                controller interfaces.

        Outputs:
            np.ndarray:
                Optimal control vector:
                    [v, omega]

        ROS:
            Publishers : None
            Subscribers: None
            Services   : None
        """

        # Projection vectors used in the CLF formulation
        proj_perp = -np.array([[-np.sin(rbt_pose[2])],
                               [np.cos(rbt_pose[2])]])

        proj_alog = -np.array([[np.cos(rbt_pose[2])],
                               [np.sin(rbt_pose[2])]])

        m_z = proj_perp.T @ (rbt_pose[0:2] - gamma_s[0:2])
        n_z = proj_alog.T @ (rbt_pose[0:2] - gamma_s[0:2])

        # Lyapunov function
        V = 0.5 * (
            self.linear_gain_sq *
            ((rbt_pose[0] - gamma_s[0]) ** 2 +
             (rbt_pose[1] - gamma_s[1]) ** 2)
            +
            self.angular_gain_sq *
            np.arctan2(m_z, n_z) ** 2
        )

        dV_dxy = (
            self.linear_gain_sq *
            (rbt_pose[0:2] - gamma_s[0:2]).T
            +
            self.angular_gain_sq *
            np.arctan2(m_z, n_z) *
            (1 / (m_z ** 2 + n_z ** 2)) *
            (n_z * proj_perp.T - m_z * proj_alog.T)
        )

        dV_dtheta = -self.angular_gain_sq * np.arctan2(m_z, n_z)

        dVdx = np.row_stack(
            (dV_dxy[0, 0], dV_dxy[0, 1], dV_dtheta)
        )

        # Unicycle dynamics using off-wheel formulation
        g_x = np.array([
            [np.cos(rbt_pose[2]), -self.l * np.sin(rbt_pose[2])],
            [np.sin(rbt_pose[2]),  self.l * np.cos(rbt_pose[2])],
            [0, 1]
        ])

        uu = cp.Variable(3)

        # CLF-QP constraints
        constraints = [
            dVdx.T @ (g_x @ uu[:2]) + self.rateV * V <= uu[2],
            uu[2] >= 0.0,
            cp.abs(uu[0]) <= self.max_v,
            cp.abs(uu[1]) <= 1.00
        ]

        # Adaptive objective weights based on safety metric
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

        # Solve CLF-QP
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
