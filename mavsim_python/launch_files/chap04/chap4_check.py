"""
mavsimPy
    Homework check for chapter 4
        1/5/2023 - David L. Christiansen
        7/13/2023 - RWB
        2025-02-12 - engband
"""

# ======================================
# ======================================

# standard lib
import os
import sys
# insert parent directory at beginning of python search path
from pathlib import Path
# sys.path.insert(0,os.fspath(Path(__file__).parents[1]))
sys.path.insert(0,os.fspath(Path(__file__).parents[2]))

# 3rd party
import numpy as np

# local
import parameters.simulation_parameters as SIM
from models.mav_dynamics_control import MavDynamics
from message_types.msg_delta import MsgDelta
from models.wind_simulation import WindSimulation

import tools.color
import tools.check_funcs as ckfns

wind = WindSimulation(SIM.ts_simulation)
mav = MavDynamics(SIM.ts_simulation)

# ======================================
# ======================================
# correct values to compare against

### 1st Case ###

# Propeller Forces and Torque
T_p_c01 = -12.43072534597213
Q_p_c01 = -0.49879620097737803

# Forces and Moments
forces_c01 = np.array([
    -21.48250762600656,
    0.20707328125000002,
    63.44373750624077
])
moments_c01 = np.array([
    0.506370113312378,
    8.75643373378125,
    -0.21774997963125006
])

# State Derivatives
xdot_ned_c01 = np.array([25.0, 0.0, 0.0])
xdot_uvw_c01 = np.array([
    -1.952955238727869,
    0.01882484375,
    5.767612500567343
])
xdot_quat_c01 = np.array([-0.0, 0, 0, 0])
xdot_pqr_c01 = np.array([
    0.6021690003674434,
    7.714919589234582,
    -0.0825746628692495
])

# ======================
### 2nd Case ###

# Propeller Forces and Torque
T_p_c02 = 37.7794805541605
Q_p_c02 =  1.809846739787848

# Forces and Moments
forces_c02 = np.array([
    27.626594049217356,
    54.13991070468528,
    46.97294443218653
])
moments_c02 = np.array([
    1.6030203500067557,
    5.982053219886495,
    -1.1805441645292776
])

# State Derivatives
xdot_ned_c02 = np.array([
    24.283238643486627,
    12.605130052025968,
    1.2957327060769266
])
xdot_uvw_c02 = np.array([
    2.3779189341423796,
    0.2308339966235602,
    8.881532282520418
])
xdot_quat_c02 = np.array([
    -0.025995661302161892,
    -0.011500703223228347,
    0.05851804333262313,
    0.10134276693843723
])
xdot_pqr_c02 = np.array([
    1.8427420637214975,
    5.2743652738342774,
    -0.5471458931221012
])

# ======================
### 3rd Case ###

# include Wind

# Wind Update
Va_c03 = 27.39323489287441
alpha_c03 = 0.05259649205640062
beta_c03 = 0.022795289526122857

# Propeller Forces and Torque
T_p_c03 = 31.31315544701058
Q_p_c03 = 1.5877828779895595

# Forces and Moments
forces_c03 = np.array([
    24.990405310886967,
    48.44244456621981,
    -39.98407127810252
])
moments_c03 = np.array([
    0.10925814573797177,
    0.1249623335264915,
    -0.0951377745644831
])

# State Derivatives
xdot_ned_c03 = np.array([
    24.283238643486627,
    12.605130052025968,
    1.2957327060769266
])
xdot_uvw_c03 = np.array([
    2.138265412475981,
    -0.2871174705096644,
    0.976349036130503
])
xdot_quat_c03 = np.array([
    -0.025995661302161892,
    -0.011500703223228347,
    0.05851804333262313,
    0.10134276693843723
])
xdot_pqr_c03 = np.array([
    0.10353614217633977,
    0.11393277483867911,
    -0.04913225070552499
])

# ======================================
# ======================================
# ======================================
# ======================================
### 1st Case ###
print(f"\n\t{tools.color.cyan("### 1st Case ###")}\n")


delta = MsgDelta()
delta.elevator = -0.2
delta.aileron = 0.0
delta.rudder = 0.005
delta.throttle = 0.5

T_p, Q_p = mav._motor_thrust_torque(mav._Va, delta.throttle)
# print("Propeller Forces and Torque", "\n")
print(f"{       "T_p":>{ckfns.lpad}}: {ckfns.ck_err(T_p_c01, T_p)}")
print(f"{       "Q_p":>{ckfns.lpad}}: {ckfns.ck_err(Q_p_c01, Q_p)}\n")

forces_moments = mav._forces_moments(delta)
# print("Forces and Moments : Case 1", "\n")
print(f"{      "fxyz":>{ckfns.lpad}}: {ckfns.ck_err(forces_c01,  forces_moments[:3, 0])}")
print(f"{      "Mxyz":>{ckfns.lpad}}: {ckfns.ck_err(moments_c01, forces_moments[3:, 0])}\n")

x_dot = mav._f(mav._state, forces_moments)
# print("State Derivatives : Case 1", "\n")
print(f"{ "x_ned_dot":>{ckfns.lpad}}: {ckfns.ck_err(xdot_ned_c01,  x_dot[:3,   0])}")
print(f"{ "x_uvw_dot":>{ckfns.lpad}}: {ckfns.ck_err(xdot_uvw_c01,  x_dot[3:6,  0])}")
print(f"{"x_quat_dot":>{ckfns.lpad}}: {ckfns.ck_err(xdot_quat_c01, x_dot[6:10, 0])}")
print(f"{ "x_ned_dot":>{ckfns.lpad}}: {ckfns.ck_err(xdot_pqr_c01,  x_dot[10:,  0])}\n")


# ======================================
# ======================================
### 2nd Case ###
print(f"\t{tools.color.cyan("### 2nd Case ###")}\n")


delta.elevator = -0.15705144
delta.aileron = 0.01788999
delta.rudder = 0.01084654
delta.throttle = 1.

mav._state = np.array([
    [ 6.19506532e+01],
    [ 2.22940203e+01],
    [-1.10837551e+02],
    [ 2.73465947e+01],
    [ 6.19628233e-01],
    [ 1.42257772e+00],
    [ 9.38688796e-01],
    [ 2.47421558e-01],
    [ 6.56821468e-02],
    [ 2.30936730e-01],
    [ 4.98772167e-03],
    [ 1.68736005e-01],
    [ 1.71797313e-01]
])

T_p, Q_p = mav._motor_thrust_torque(mav._Va, delta.throttle)
# print("Propeller Forces and Torque", "\n")
print(f"{       "T_p":>{ckfns.lpad}}: {ckfns.ck_err(T_p_c02, T_p)}")
print(f"{       "Q_p":>{ckfns.lpad}}: {ckfns.ck_err(Q_p_c02, Q_p)}\n")

forces_moments = mav._forces_moments(delta)
# print("Forces and Moments : Case 2" , "\n")
print(f"{      "fxyz":>{ckfns.lpad}}: {ckfns.ck_err(forces_c02,    forces_moments[:3, 0])}")
print(f"{      "Mxyz":>{ckfns.lpad}}: {ckfns.ck_err(moments_c02,   forces_moments[3:, 0])}\n")

x_dot = mav._f(mav._state, forces_moments)
# print("State Derivatives : Case 2", "\n")
print(f"{ "x_ned_dot":>{ckfns.lpad}}: {ckfns.ck_err(xdot_ned_c02,  x_dot[:3,   0])}")
print(f"{ "x_uvw_dot":>{ckfns.lpad}}: {ckfns.ck_err(xdot_uvw_c02,  x_dot[3:6,  0])}")
print(f"{"x_quat_dot":>{ckfns.lpad}}: {ckfns.ck_err(xdot_quat_c02, x_dot[6:10, 0])}")
print(f"{ "x_ned_dot":>{ckfns.lpad}}: {ckfns.ck_err(xdot_pqr_c02,  x_dot[10:,  0])}\n")


# ======================================
# ======================================
### 3rd Case ###
print(f"\t{tools.color.cyan("### 3rd Case ###")}\n")


current_wind = np.array([
    [ 0.        ],
    [ 0.        ],
    [ 0.        ],
    [-0.00165177],
    [-0.00475441],
    [-0.01717199]
])

mav._update_velocity_data(current_wind)
# print("Wind Update" , "\n")
print(f"{        "Va":>{ckfns.lpad}}: {ckfns.ck_err(Va_c03,    mav._Va)}")
print(f"{     "alpha":>{ckfns.lpad}}: {ckfns.ck_err(alpha_c03, mav._alpha)}")
print(f"{      "beta":>{ckfns.lpad}}: {ckfns.ck_err(beta_c03,  mav._beta)}\n")

T_p, Q_p = mav._motor_thrust_torque(mav._Va, delta.throttle)
# print("Propeller Forces and Torque", "\n")
print(f"{       "T_p":>{ckfns.lpad}}: {ckfns.ck_err(T_p_c03, T_p)}")
print(f"{       "Q_p":>{ckfns.lpad}}: {ckfns.ck_err(Q_p_c03, Q_p)}\n")

forces_moments = mav._forces_moments(delta)
# print("Forces and Moments : Case w/Wind" , "\n")
print(f"{      "fxyz":>{ckfns.lpad}}: {ckfns.ck_err(forces_c03,    forces_moments[:3, 0])}")
print(f"{      "Mxyz":>{ckfns.lpad}}: {ckfns.ck_err(moments_c03,   forces_moments[3:, 0])}\n")

x_dot = mav._f(mav._state, forces_moments)
# print("State Derivatives : Case w/Wind", "\n")
print(f"{ "x_ned_dot":>{ckfns.lpad}}: {ckfns.ck_err(xdot_ned_c03,  x_dot[:3,   0])}")
print(f"{ "x_uvw_dot":>{ckfns.lpad}}: {ckfns.ck_err(xdot_uvw_c03,  x_dot[3:6,  0])}")
print(f"{"x_quat_dot":>{ckfns.lpad}}: {ckfns.ck_err(xdot_quat_c03, x_dot[6:10, 0])}")
print(f"{ "x_ned_dot":>{ckfns.lpad}}: {ckfns.ck_err(xdot_pqr_c03,  x_dot[10:,  0])}\n")
