"""
mavsimPy
    Homework check for chapter 3
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
sys.path.insert(0,os.fspath(Path(__file__).parents[2]))

# 3rd party
import numpy as np

# local
from models.mav_dynamics import MavDynamics
import parameters.simulation_parameters as SIM
import parameters.aerosonde_parameters as MAV

import tools.color
import tools.check_funcs as ckfns

# ======================================
# ======================================
# correct values to compare against

### 1st Case ###

xdot_ned_c01 = np.array([5.0, 0.0, 0.0])
xdot_uvw_c01 = np.array([
    0.90909091,
    0.45454545,
    2.5
])
xdot_quat_c01 = np.array([-0.0, 0.5, 0.25, 0])
xdot_pqr_c01 = np.array([
    0.06073576,
    12.22872247,
    -0.08413156
])

# ======================
### 2nd Case ###

xdot_ned_c02 = np.array([
    0.08746356,
    -1.96793003,
    2.79883382
])
xdot_uvw_c02 = np.array([
    9.90909091,
    0.45454545,
    0.
])
xdot_quat_c02 = np.array([
    -0.3,
    0.,
    -0.9,
    1.5
])
xdot_pqr_c02 = np.array([
    0.,
    13.28951542,
    0.
])

# ======================================
# ======================================
# ======================================
# ======================================
### 1st Case ###
print(f"\n\t{tools.color.cyan("### 1st Case ###")}\n")


state = np.array([[
    5, 2, -20,      # ned
    5, 0, 0,        # uvw
    1, 0, 0, 0,     # quat
    1, 0.5, 0,      # pqr
    0, 0            #
]]).T
forces_moments = np.array([[
    10, 5, 0,
    0, 14, 0
]]).T
mav = MavDynamics(SIM.ts_simulation)
x_dot = mav._f(state, forces_moments)

# print("State Derivatives: Case 1")
print(f"{ "x_ned_dot":>{ckfns.lpad}}: {ckfns.ck_err(xdot_ned_c01,  x_dot[:3,   0])}")
print(f"{ "x_uvw_dot":>{ckfns.lpad}}: {ckfns.ck_err(xdot_uvw_c01,  x_dot[3:6,  0])}")
print(f"{"x_quat_dot":>{ckfns.lpad}}: {ckfns.ck_err(xdot_quat_c01, x_dot[6:10, 0])}")
print(f"{ "x_ned_dot":>{ckfns.lpad}}: {ckfns.ck_err(xdot_pqr_c01,  x_dot[10:,  0])}\n")


# ======================================
# ======================================
### 2nd Case ###
print(f"\t{tools.color.cyan("### 2nd Case ###")}\n")

state = np.array([[
    5, 2, -20,
    0, 3, 6,
    1, .6, 0, .2,
    0, 0, 3,
    0, 0
]]).T
forces_moments = np.array([[
    10, 5, 0,
    0, 14, 0
]]).T
mav = MavDynamics(SIM.ts_simulation)
x_dot = mav._f(state, forces_moments)

# print("State Derivatives: Case 2")
print(f"{ "x_ned_dot":>{ckfns.lpad}}: {ckfns.ck_err(xdot_ned_c02,  x_dot[:3,   0])}")
print(f"{ "x_uvw_dot":>{ckfns.lpad}}: {ckfns.ck_err(xdot_uvw_c02,  x_dot[3:6,  0])}")
print(f"{"x_quat_dot":>{ckfns.lpad}}: {ckfns.ck_err(xdot_quat_c02, x_dot[6:10, 0])}")
print(f"{ "x_ned_dot":>{ckfns.lpad}}: {ckfns.ck_err(xdot_pqr_c02,  x_dot[10:,  0])}\n")
