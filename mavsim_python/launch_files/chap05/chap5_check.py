"""
mavsimPy
    Homework check for chapter 5
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
# sys.path.insert(0,os.fspath(Path(__file__).parents[2]))

# 3rd party
import numpy as np

# local
import models.mav_dynamics_control
import models.compute_models

from models.mav_dynamics_control import MavDynamics
from models.compute_models import compute_ss_model, compute_tf_model, euler_state, quaternion_state, f_euler, df_dx, df_du, dT_dVa, dT_ddelta_t
import parameters.simulation_parameters as SIM
from message_types.msg_delta import MsgDelta

import tools.color
import tools.check_funcs as ckfns

# ======================================
# ======================================
# correct values to compare against

### 1st Case ###

A_lon_c01 = np.array([
    [-0.27486079,  0.49868179, -1.21983882, -9.79511927, -0.0],
    [-0.56234374, -4.49810469, 24.37105023, -0.53938541, -0.0],
    [ 0.19993539, -3.99297865, -5.29473836,  0.        , -0.0],
    [ 0.        ,  0.        ,  0.99997406,  0.        , -0.0],
    [ 0.04999035, -0.9987497 , -0.        , 24.99958361,  0.0]
])

B_lon_c01 = np.array([
    [ -0.13840016,   8.20722086],
    [ -2.58618345,   0.0       ],
    [-36.11239041,   0.0       ],
    [  0.0,          0.0       ],
    [ -0.0,         -0.0       ]
])

A_lat_c01 = np.array([
    [-7.76772566e-01,    1.24975500e+00, -2.49687430e+01,  9.79757127e+00,  0.0],
    [-3.86671904e+00,   -2.26288510e+01,  1.09050409e+01,  0.0,             0.0],
    [ 7.83077082e-01,   -1.15091678e-01, -1.22765475e+00,  0.0,             0.0],
    [ 0.0,               9.99999666e-01,  5.00528958e-02,  0.0,             0.0],
    [ 0.0,              -1.67051761e-08,  1.00125153e+00,  0.0,             0.0]
])

B_lat_c01 = np.array([
    [  1.48617191,   3.76496884],
    [130.88368125,  -1.79637441],
    [  5.01173513, -24.88134191],
    [  0.0,          0.0       ],
    [  0.0,          0.0       ]
])

Va_trim_c01     =  25.000000291201477
alpha_trim_c01  =   0.05001104395214544
theta_trim_c01  =   0.05001119284259128
a_phi1_c01      =  22.62885095683996
a_phi2_c01      = 130.88368124853207
a_theta1_c01    =   5.294738359662443
a_theta2_c01    =  99.94742395724161
a_theta3_c01    = -36.11239040790846
a_V1_c01        =   0.2888454121899283
a_V2_c01        =   8.20722086381344
a_V3_c01        =   9.809999999999892
comp_tf_mod_01 = np.array([
    Va_trim_c01,    alpha_trim_c01,     theta_trim_c01,
    a_phi1_c01,     a_phi2_c01,
    a_theta1_c01,   a_theta2_c01,       a_theta3_c01,
    a_V1_c01,       a_V2_c01,           a_V3_c01
])

x_euler__c01 = np.array([[
    0.0,            -0.0,               -1.00000000e+02,
    2.49687430e+01,  0.0,                1.24975500e+00,
    0.0,             5.00111928e-02,     0.0,
    0.0,             0.0,                0.0
]]).T

x_quat__c01 = np.array([[
    0.0,            -0.0,   -1.00000000e+02,
    2.49687430e+01,  0.0,    1.24975500e+00,
    9.99687376e-01,  0.0,    2.50029906e-02,    0.0,
    0.0,             0.0,    0.0
]]).T

f_euler__c01 = np.array([[
     2.50000003e+01,     0.0,               -3.72226119e-06,
    -8.51519325e-01,     1.58782607e-03,    -3.26078187e-02,
     0.0,                0.0,                0.0,
    -4.98388573e-05,    -1.47473075e-06,     2.51707628e-04
]]).T

A__c01 = np.array([
    [ 0.0,               0.0,                0.0,                9.98749701e-01,
      0.0,               4.99903481e-02,    -3.12375834e-04,    -1.25002682e-01,
     -1.24998960e-01,    0.0,                0.0,                0.0],
    [ 0.0,               0.0,                0.0,                0.0,
      1.00000000e+00,    0.0,               -1.24973417e+00,     0.0,
      2.49995836e+01,    0.0,                0.0,                0.0],
    [ 0.0,               0.0,                0.0,               -4.99903481e-02,
      0.0,               9.98749701e-01,    -6.24091015e-03,    -2.49995836e+01,
      1.38750773e-14,    0.0,                0.0,                0.0],
    [ 0.0,               0.0,                0.0,               -2.74860789e-01,
     -4.99239248e-05,    4.98681787e-01,     0.0,               -9.79511927e+00,
      0.0,               0.0,               -1.21983882e+00,     0.0],
    [ 0.0,               0.0,                0.0,                1.26892670e-04,
     -7.76772566e-01,    6.37546449e-06,     9.79757127e+00,     0.0,
      0.0,               1.24975500e+00,     0.0,               -2.49687430e+01],
    [ 0.0,               0.0,                0.0,               -5.62343736e-01,
     -1.57285474e-04,   -4.49810469e+00,    -4.89882646e-02,    -5.39385406e-01,
      1.29063427e-13,    0.0,                2.43710502e+01,     0.0],
    [ 0.0,               0.0,                0.0,                0.0,
      0.0,               0.0,                0.0,                0.0,
      0.0,               9.99999666e-01,     0.0,                5.00528958e-02],
    [ 0.0,               0.0,                0.0,                0.0,
      0.0,               0.0,                0.0,                0.0,
      0.0,              -2.43928799e-05,     9.99974056e-01,    -2.56443516e-05],
    [ 0.0,               0.0,                0.0,                0.0,
      0.0,               0.0,                0.0,                0.0,
      0.0,              -1.67051761e-08,     0.0,                1.00125153e+00],
    [ 0.0,               0.0,                0.0,                1.43288661e-01,
     -3.86671904e+00,    7.19895526e-03,     0.0,                0.0,
      0.0,              -2.26288510e+01,     0.0,                1.09050409e+01],
    [ 0.0,               0.0,                0.0,                1.99935388e-01,
     -2.35956915e-11,   -3.99297865e+00,     0.0,                0.0,
      0.0,              -1.06079295e-03,    -5.29473836e+00,     1.06079295e-03],
    [ 0.0,               0.0,                0.0,                9.82820776e-03,
      7.83077082e-01,    4.93778316e-04,     0.0,                0.0,
      0.0,              -1.15091678e-01,     0.0,               -1.22765475e+00]
])

B__c01 = np.array([
    [ 0.0,              0.0,             0.0,                0.0],
    [ 0.0,              0.0,             0.0,                0.0],
    [ 0.0,              0.0,             0.0,                0.0],
    [-1.38400158e-01,   0.0,             0.0,                8.20722086e+00],
    [ 0.0,              1.48617191e+00,  3.76496884e+00,     1.30104261e-16],
    [-2.58618345e+00,   0.0,             0.0,                0.0],
    [ 0.0,              0.0,             0.0,                0.0],
    [ 0.0,              0.0,             0.0,                0.0],
    [ 0.0,              0.0,             0.0,                0.0],
    [ 0.0,              1.30883681e+02, -1.79637441e+00,    -5.31386382e+00],
    [-3.61123904e+01,   0.0,             0.0,                0.0],
    [ 0.0,              5.01173513e+00, -2.48813419e+01,    -3.63723254e-01]
])

dT_dVa__c01         = -2.3521982958853327
dT_ddelta_t__c01    = 90.27942950194785

# ======================
### 2nd Case ###

x_euler__c02 = np.array([[
    6.19506532e+01,     2.22940203e+01,     -1.10837551e+02,
    2.73465947e+01,     6.19628233e-01,      1.42257772e+00,
    5.17674540e-01,     9.03286236e-03,      4.84851312e-01,
    4.98772167e-03,     1.68736005e-01,      1.71797313e-01
]]).T

x_quat__c02 = np.array([[
    6.19506532e+01,     2.22940203e+01,     -1.10837551e+02,
    2.73465947e+01,     6.19628233e-01,      1.42257772e+00,
    9.38688796e-01,     2.47421558e-01,      6.56821468e-02,    2.30936730e-01,
    4.98772167e-03,     1.68736005e-01,      1.71797313e-01
]]).T

f_euler__c02 = np.array([[
     2.42832387e+01,     1.26051301e+01,    1.29573271e+00,
    -2.33922378e+00,    -3.41372912e-01,    1.19547839e+00,
     7.06527544e-03,     6.16229478e-02,    2.32773972e-01,
     2.20713174e-01,     2.05034334e+00,    2.11992826e-01
]]).T

A__c02 = np.array([
    [ 0.0,               0.0,                0.0,                8.84708165e-01,
     -4.01053084e-01,    2.37587641e-01,     7.17285818e-01,     1.02534355e+00,
     -1.27263352e+01,    0.0,                0.0,                0.0],
    [ 0.0,               0.0,                0.0,                4.66057800e-01,
      7.70901599e-01,   -4.34166848e-01,    -1.36496690e+00,     5.40143494e-01,
      2.42198088e+01,    0.0,                0.0,                0.0],
    [ 0.0,               0.0,                0.0,               -9.03273952e-03,
      4.94840529e-01,    8.68936857e-01,    -1.73242216e-01,    -2.73654375e+01,
      8.88178420e-14,    0.0,                0.0,                0.0],
    [ 0.0,               0.0,                0.0,               -2.81015715e-01,
      1.66057646e-01,    3.90343881e-01,     8.88178420e-14,    -9.80899325e+00,
      8.88178420e-14,    0.0,               -1.38851585e+00,     6.19628233e-01],
    [ 0.0,               0.0,                0.0,               -1.89369528e-01,
     -8.51679834e-01,    4.07039158e-03,     8.49985677e+00,    -6.81211843e-02,
      0.0,               1.42257772e+00,     0.0,               -2.73465947e+01],
    [ 0.0,               0.0,                0.0,               -4.43677505e-01,
     -2.47775268e-02,   -4.92908300e+00,    -4.89692568e+00,    -1.19620371e-01,
      1.33226763e-13,   -6.19628233e-01,     2.66918142e+01,     0.0],
    [ 0.0,               0.0,                0.0,                0.0,
      0.0,               0.0,                4.57332278e-04,     2.32785596e-01,
      6.84717270e-06,    9.99803542e-01,     4.65945661e-03,     7.52232695e-03],
    [ 0.0,               0.0,                0.0,                0.0,
      0.0,               0.0,               -2.33045276e-01,    -1.45951852e-04,
      2.88111478e-05,    9.72629246e-05,     8.68868663e-01,    -4.94693211e-01],
    [ 0.0,               0.0,                0.0,                0.0,
      0.0,               0.0,                6.04210285e-02,     3.21737153e-03,
     -4.96668711e-05,   -1.81663545e-04,     4.95056161e-01,     8.68704383e-01],
    [ 0.0,               0.0,                0.0,                1.22909150e-01,
     -4.23482873e+00,    6.41465012e-03,     0.0,                0.0,
      0.0,              -2.47721978e+01,    -1.32477696e-01,     1.18171038e+01],
    [ 0.0,               0.0,                0.0,                4.12603256e-01,
      4.22266864e-03,   -4.36582619e+00,     0.0,                0.0,
      0.0,               1.39345132e-01,    -5.80103825e+00,     4.16161389e-02],
    [ 0.0,               0.0,                0.0,                1.02442961e-02,
      8.58419490e-01,    5.34772009e-04,     0.0,                0.0,
      0.0,              -1.54489161e-01,    -2.17077302e-02,    -1.36554366e+00]
])

B__c02 = np.array([
    [  0.0,          0.0,          0.0,          0.0       ],
    [  0.0,          0.0,          0.0,          0.0       ],
    [  0.0,          0.0,          0.0,          0.0       ],
    [ -0.16004176,   0.0,          0.0,          5.51743963],
    [  0.0,          1.78398623,   4.51943179,   0.0       ],
    [ -3.10474937,   0.0,          0.0,          0.0       ],
    [  0.0,          0.0,          0.0,          0.0       ],
    [  0.0,          0.0,          0.0,          0.0       ],
    [  0.0,          0.0,          0.0,          0.0       ],
    [  0.0,        157.1114916,   -2.15635028,  -4.51605243],
    [-43.34896045,   0.0,          0.0,          0.0       ],
    [  0.0,          6.01603786, -29.86731962,  -0.30911467]
])

dT_dVa__c02         = -2.3737948095355677
dT_ddelta_t__c02    = 60.69183591548715

# ======================================
# ======================================
# ======================================
# ======================================
### 1st Case ###
print(f"\n\t{tools.color.cyan("### 1st Case ###")}\n")


mav = MavDynamics(SIM.ts_simulation)

trim_state = np.array([[
     0.0,       -0.0,  -100.0,
    24.968743,   0.0,     1.249755,
     0.999687,   0.0,     0.025003,    0.0,
     0.0,        0.0,     0.0
]]).T
trim_input = MsgDelta(
    elevator=-0.124778,
    aileron=0.001836,
    rudder=-0.000303,
    throttle=0.676752
)
mav._state = trim_state

A_lon, B_lon, A_lat, B_lat = compute_ss_model(mav, trim_state, trim_input)

A_lon_c01_res = ckfns.ck_err(A_lon_c01, A_lon)
B_lon_c01_res = ckfns.ck_err(B_lon_c01, B_lon)
A_lat_c01_res = ckfns.ck_err(A_lat_c01, A_lat)
B_lat_c01_res = ckfns.ck_err(B_lat_c01, B_lat)

print(f"{ "A_lon":>{ckfns.lpad}}: {A_lon_c01_res}")
print(f"{ "B_lon":>{ckfns.lpad}}: {B_lon_c01_res}")
print(f"{ "A_lat":>{ckfns.lpad}}: {A_lat_c01_res}")
print(f"{ "B_lat":>{ckfns.lpad}}: {B_lat_c01_res}\n")

Va_trim, alpha_trim, theta_trim, a_phi1, a_phi2, a_theta1, a_theta2, a_theta3, \
    a_V1, a_V2, a_V3 = compute_tf_model(mav, trim_state, trim_input)

comp_tf_mod_check = np.array([
    Va_trim,     alpha_trim,    theta_trim,
    a_phi1,      a_phi2,
    a_theta1,    a_theta2,      a_theta3,
    a_V1,        a_V2,          a_V3
])

# comp_tf_mod_idxs = ckfns.ck_err(comp_tf_mod_01, comp_tf_mod_check)
# print(f"compute_tf_model: {comp_tf_mod_idxs}")

print(f"{   "Va_trim":>{ckfns.lpad}}: {ckfns.ck_err(Va_trim_c01,      Va_trim)}")
print(f"{"alpha_trim":>{ckfns.lpad}}: {ckfns.ck_err(alpha_trim_c01,   alpha_trim)}")
print(f"{"theta_trim":>{ckfns.lpad}}: {ckfns.ck_err(theta_trim_c01,   theta_trim)}")
print(f"{    "a_phi1":>{ckfns.lpad}}: {ckfns.ck_err(a_phi1_c01,       a_phi1)}")
print(f"{    "a_phi2":>{ckfns.lpad}}: {ckfns.ck_err(a_phi2_c01,       a_phi2)}")
print(f"{  "a_theta1":>{ckfns.lpad}}: {ckfns.ck_err(a_theta1_c01,     a_theta1)}")
print(f"{  "a_theta2":>{ckfns.lpad}}: {ckfns.ck_err(a_theta2_c01,     a_theta2)}")
print(f"{  "a_theta3":>{ckfns.lpad}}: {ckfns.ck_err(a_theta3_c01,     a_theta3)}")
print(f"{      "a_V1":>{ckfns.lpad}}: {ckfns.ck_err(a_V1_c01,         a_V1)}")
print(f"{      "a_V2":>{ckfns.lpad}}: {ckfns.ck_err(a_V2_c01,         a_V2)}")
print(f"{      "a_V3":>{ckfns.lpad}}: {ckfns.ck_err(a_V3_c01,         a_V3)}\n")

x_euler_            = euler_state(trim_state)
x_euler__c01_res    = ckfns.ck_err(x_euler__c01, x_euler_)

x_quat_             = quaternion_state(x_euler_)
x_quat__c01_res     = ckfns.ck_err(x_quat__c01, x_quat_)

f_euler_            = f_euler(mav, x_euler_, trim_input)
f_euler__c01_res    = ckfns.ck_err(f_euler__c01, f_euler_)

A_                  = df_dx(mav, x_euler_, trim_input)
A__c01_res          = ckfns.ck_err(A__c01, A_)

B_                  = df_du(mav, x_euler_, trim_input)
B__c01_res          = ckfns.ck_err(B__c01, B_)

dT_dVa_             =  dT_dVa(mav, mav._Va, trim_input.throttle)
dT_dVa__c01_res     = ckfns.ck_err(dT_dVa__c01, dT_dVa_)

dT_ddelta_t_        =  dT_ddelta_t(mav, mav._Va, trim_input.throttle)
dT_ddelta_t__c01_res = ckfns.ck_err(dT_ddelta_t__c01, dT_ddelta_t_)

print(f"{    "x_euler_":>{ckfns.lpad}}: {x_euler__c01_res}")
print(f"{     "x_quat_":>{ckfns.lpad}}: {x_quat__c01_res}")
print(f"{    "f_euler_":>{ckfns.lpad}}: {f_euler__c01_res}\n")
print(f"{          "A_":>{ckfns.lpad}}: {A__c01_res}")
print(f"{          "B_":>{ckfns.lpad}}: {B__c01_res}\n")
print(f"{     "dT_dVa_":>{ckfns.lpad}}: {dT_dVa__c01_res}")
print(f"{"dT_ddelta_t_":>{ckfns.lpad}}: {dT_ddelta_t__c01_res}\n")

# ======================================
# ======================================
### 2nd Case ###
print(f"\t{tools.color.cyan("### 2nd Case ###")}\n")

mav._state = np.array([[
    6.19506532e+01,     2.22940203e+01,     -1.10837551e+02,
    2.73465947e+01,     6.19628233e-01,      1.42257772e+00,
    9.38688796e-01,     2.47421558e-01,      6.56821468e-02,    2.30936730e-01,
    4.98772167e-03,     1.68736005e-01,      1.71797313e-01
]]).T

mav._Va = 22.4
new_state = mav._state
delta = MsgDelta()
delta.elevator = -0.2
delta.aileron = 0.0
delta.rudder = 0.005
delta.throttle = 0.5

x_euler_c2 = euler_state(new_state)
x_euler__c02_res = ckfns.ck_err(x_euler__c02, x_euler_c2)

x_quat_c2 = quaternion_state(x_euler_c2)
x_quat__c02_res = ckfns.ck_err(x_quat__c02, x_quat_c2)

f_euler_c2 = f_euler(mav, x_euler_c2, delta)
f_euler__c02_res = ckfns.ck_err(f_euler__c02, f_euler_c2)

A_c2 = df_dx(mav, x_euler_c2, delta)
A__c02_res = ckfns.ck_err(A__c02, A_c2)

B_c2 = df_du(mav, x_euler_c2, delta)
B__c02_res = ckfns.ck_err(B__c02, B_c2)

dT_dVa_c2 =  dT_dVa(mav, mav._Va, delta.throttle)
dT_dVa__c02_res = ckfns.ck_err(dT_dVa__c02, dT_dVa_c2)

dT_ddelta_t_c2 =  dT_ddelta_t(mav, mav._Va, delta.throttle)
dT_ddelta_t__c02_res = ckfns.ck_err(dT_ddelta_t__c02, dT_ddelta_t_c2)

print(f"{     "x_euler":>{ckfns.lpad}}: {x_euler__c02_res}")
print(f"{      "x_quat":>{ckfns.lpad}}: {x_quat__c02_res}")
print(f"{     "f_euler":>{ckfns.lpad}}: {f_euler__c02_res}\n")
print(f"{           "A":>{ckfns.lpad}}: {A__c02_res}")
print(f"{           "B":>{ckfns.lpad}}: {B__c02_res}\n")
print(f"{      "dT_dVa":>{ckfns.lpad}}: {dT_dVa__c02_res}")
print(f"{ "dT_ddelta_t":>{ckfns.lpad}}: {dT_ddelta_t__c02_res}\n")
