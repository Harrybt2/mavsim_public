"""
mavsimPy
    Homework check for chapter 7
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
# sys.path.insert(0,os.fspath(Path(__file__).parents[2]))

# 3rd party
import numpy as np

# local
import parameters.simulation_parameters as SIM
from models.mav_dynamics_sensors import MavDynamics
from message_types.msg_delta import MsgDelta
from models.wind_simulation import WindSimulation

import tools.color
import tools.check_funcs as ckfns

# ======================================
# ======================================
# correct values to compare against

### 1st Case ###

# Sensor Measurments
gyro_xyz_c01 = np.array([
    0.01650379218706677,
    0.06427042225632044,
    0.021514085435896575
])
accel_xyz_c01 = np.array([
    -1.9179064575668672,
    0.021999076875292966,
    -4.09051174211739
])
mag_xyz_c01 = np.array([
    0.3971476016145697,
    -0.21652924195767995,
    -0.8917997627497268
])

abs_pressure_c01 = 1232.0791418164015
diff_pressure_c01 = 400.01451351359765

gps_neh_Vg_course_c01 = np.array([
    0.22949188981847735,
    -0.006903273537024948,
    100.00618637424958,
    24.984960099451566,
    0.00016841816774766126
])

# ======================
### 2nd Case ###

# Sensor Measurments
gyro_xyz_c02 = np.array([
    -0.11242925086884417,
    0.6817844242237464,
    -0.0978644004322639
])
accel_xyz_c02 = np.array([
    0.9883828280371655,
    0.15627594214667928,
    -49.75496744909434
])
mag_xyz_c02 = np.array([
    0.7520934576699337,
    0.4276210842005712,
    -0.5014562525860837
])

abs_pressure_c02 = 540.1184916254514
diff_pressure_c02 = 2637.0762242313144

gps_neh_Vg_course_c02 = np.array([
    93.46022621939377,
    -16.459729859591008,
    39.87034483806606,
    60.459194487829244,
    0.04996459981341533
])

# ======================================
# ======================================
# ======================================
# ======================================
### 1st Case ###
print(f"\n\t{tools.color.cyan("### 1st Case ###")}\n")


wind = WindSimulation(SIM.ts_simulation)
current_wind = wind.update()
mav = MavDynamics(SIM.ts_simulation)
delta = MsgDelta()
delta.elevator = -0.2
delta.aileron = 0.0
delta.rudder = 0.005
delta.throttle = 0.5
mav.update(delta, current_wind)
sensors = mav.sensors()

# print("Sensor Measurments: Case 1")
gyro_xyz = np.array([sensors.gyro_x, sensors.gyro_y, sensors.gyro_z])
accel_xyz = np.array([sensors.accel_x, sensors.accel_y, sensors.accel_z])
mag_xyz = np.array([sensors.mag_x, sensors.mag_y, sensors.mag_z])
gps_neh_Vg_course = np.array([sensors.gps_n, sensors.gps_e, sensors.gps_h, sensors.gps_Vg, sensors.gps_course])

print(f"{          "gyro_xyz":>{ckfns.lpad}}: {ckfns.ck_err(gyro_xyz_c01,          gyro_xyz)}")
print(f"{         "accel_xyz":>{ckfns.lpad}}: {ckfns.ck_err(accel_xyz_c01,         accel_xyz)}")
print(f"{           "mag_xyz":>{ckfns.lpad}}: {ckfns.ck_err(mag_xyz_c01,           mag_xyz)}")
print(f"{      "abs_pressure":>{ckfns.lpad}}: {ckfns.ck_err(abs_pressure_c01,      sensors.abs_pressure)}")
print(f"{     "diff_pressure":>{ckfns.lpad}}: {ckfns.ck_err(diff_pressure_c01,     sensors.diff_pressure)}")
print(f"{ "gps_neh_Vg_course":>{ckfns.lpad}}: {ckfns.ck_err(gps_neh_Vg_course_c01, gps_neh_Vg_course)}\n")

# ======================================
# ======================================
### 2nd Case ###
print(f"\t{tools.color.cyan("### 2nd Case ###")}\n")


delta.elevator = -0.1
delta.aileron = 0.0
delta.rudder = 0.0
delta.throttle = 2
for i in range(1000):
    mav.update(delta, current_wind)
    mav.sensors()
#
sensors = mav.sensors()

# print("Sensor Measurments: Case 2")
gyro_xyz = np.array([sensors.gyro_x, sensors.gyro_y, sensors.gyro_z])
accel_xyz = np.array([sensors.accel_x, sensors.accel_y, sensors.accel_z])
mag_xyz = np.array([sensors.mag_x, sensors.mag_y, sensors.mag_z])
gps_neh_Vg_course = np.array([sensors.gps_n, sensors.gps_e, sensors.gps_h, sensors.gps_Vg, sensors.gps_course])

print(f"{          "gyro_xyz":>{ckfns.lpad}}: {ckfns.ck_err(gyro_xyz_c02,          gyro_xyz)}")
print(f"{         "accel_xyz":>{ckfns.lpad}}: {ckfns.ck_err(accel_xyz_c02,         accel_xyz)}")
print(f"{           "mag_xyz":>{ckfns.lpad}}: {ckfns.ck_err(mag_xyz_c02,           mag_xyz)}")
print(f"{      "abs_pressure":>{ckfns.lpad}}: {ckfns.ck_err(abs_pressure_c02,      sensors.abs_pressure)}")
print(f"{     "diff_pressure":>{ckfns.lpad}}: {ckfns.ck_err(diff_pressure_c02,     sensors.diff_pressure)}")
print(f"{ "gps_neh_Vg_course":>{ckfns.lpad}}: {ckfns.ck_err(gps_neh_Vg_course_c02, gps_neh_Vg_course)}\n")
