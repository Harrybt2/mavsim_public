"""
mavsimPy
    Homework check for chapter 7
    - Uses hard-coded expected outputs (no equation derivations shown)
"""

# standard lib
import os
import sys
from pathlib import Path
from contextlib import contextmanager

# insert parent directory at beginning of python search path
sys.path.insert(0, os.fspath(Path(__file__).parents[2]))


import numpy as np
from models.mav_dynamics_sensors import MavDynamics
import parameters.simulation_parameters as SIM
import parameters.sensor_parameters as SENSOR

import tools.color

try:
    import tools.check_funcs as ckfns
except Exception:  # pragma: no cover
    import check_funcs as ckfns


# Fix gyro biases so check values are stable across runs.
SENSOR.gyro_x_bias = 0.0123
SENSOR.gyro_y_bias = -0.0234
SENSOR.gyro_z_bias = 0.0345


# Case 1 hard-coded expected values
C1_GYRO = np.array([0.01728772167, 0.145336005, 0.206297313])
C1_ACCEL = np.array([1.8613384549683762, -4.472567411450773, -13.687906928466547])
C1_MAG = np.array([0.25849634994797613, -0.7674536708431557, -0.5866809210515955])
C1_PRESSURE = np.array([1378.934627173125, 441.942336])
C1_GPS_POS = np.array([62.18061139090499, 22.154045762927838, 110.91753646292348])
# allow two common Vg/course implementations
C1_GPS_VG_CHI_A = np.array([27.390580646341366, 0.47880088987679095])
C1_GPS_VG_CHI_B = np.array([28.140817597490567, 0.4056348532069432])
C1_ETA = np.array([0.22995819090498787, -0.13997453707216268, 0.07998546292348482])
C1_T_GPS = 0.0

# Case 2 hard-coded expected values (GPS hold branch)
C2_GYRO = np.array([-0.0977, 0.0466, 0.0145])
C2_ACCEL = np.array([-0.7318822079925629, -2.1808453915915304, -13.886281078164445])
C2_MAG = np.array([0.17484122314534903, -0.5246869258758003, -0.833147147695853])
C2_PRESSURE = np.array([1293.868368, 356.167629])
C2_GPS_POS = C1_GPS_POS.copy()  # hold last GPS sample
C2_GPS_VG_CHI_A = C1_GPS_VG_CHI_A.copy()  # hold last GPS sample
C2_GPS_VG_CHI_B = C1_GPS_VG_CHI_B.copy()  # hold last GPS sample (alt impl)
C2_ETA = C1_ETA.copy()  # hold random-walk state
C2_T_GPS = 0.01


@contextmanager
def _noise_as_mean():
    """Force np.random.normal(...) to return its mean (loc)."""
    orig_normal = np.random.normal

    def _mean_normal(loc=0.0, scale=1.0, size=None):
        if size is None:
            return float(loc)
        return np.full(size, float(loc))

    np.random.normal = _mean_normal
    try:
        yield
    finally:
        np.random.normal = orig_normal


def _meas_groups(meas):
    return {
        "gyro": np.array([meas.gyro_x, meas.gyro_y, meas.gyro_z]),
        "accel": np.array([meas.accel_x, meas.accel_y, meas.accel_z]),
        "mag": np.array([meas.mag_x, meas.mag_y, meas.mag_z]),
        "pressure": np.array([meas.abs_pressure, meas.diff_pressure]),
        "gps_pos": np.array([meas.gps_n, meas.gps_e, meas.gps_h]),
        "gps_vg_chi": np.array([meas.gps_Vg, meas.gps_course]),
    }


def _ck_err_either(soln_a, soln_b, check):
    """PASS if check matches either candidate solution."""
    tol = 1e-4
    pass_a = np.allclose(check, soln_a, rtol=tol, atol=1e-12)
    pass_b = np.allclose(check, soln_b, rtol=tol, atol=1e-12)
    if pass_a or pass_b:
        return f"{tools.color.green('PASS')}"
    return (
        f"{tools.color.red('FAIL')}  "
        f"yours = {tools.color.red(check)}  "
        f"expected one of = {tools.color.violet(soln_a)} OR {tools.color.violet(soln_b)}"
    )


mav = MavDynamics(SIM.ts_simulation)

# ----------------------
# Case 1: GPS update branch
print(f"\n\t{tools.color.cyan('### Case 1: GPS update ###')}\n")

mav._state = np.array([
    [61.9506532],
    [22.2940203],
    [-110.837551],
    [27.3465947],
    [0.619628233],
    [1.42257772],
    [0.938688796],
    [0.247421558],
    [0.0656821468],
    [0.23093673],
    [0.00498772167],
    [0.168736005],
    [0.171797313],
])
mav._forces = np.array([[19.5], [4.2], [-56.8]])
mav._Va = 26.4
mav._wind = np.array([[2.5], [-1.2], [0.4]])
mav._gps_eta_n = 0.23
mav._gps_eta_e = -0.14
mav._gps_eta_h = 0.08
mav._t_gps = SENSOR.ts_gps

with _noise_as_mean():
    meas_case1 = mav.sensors()
m1 = _meas_groups(meas_case1)

print(f"{'gyro':>{ckfns.lpad}}: {ckfns.ck_err(C1_GYRO, m1['gyro'])}")
print(f"{'accel':>{ckfns.lpad}}: {ckfns.ck_err(C1_ACCEL, m1['accel'])}")
print(f"{'mag':>{ckfns.lpad}}: {_ck_err_either(C1_MAG, np.zeros(3), m1['mag'])}")
print(f"{'pressure':>{ckfns.lpad}}: {ckfns.ck_err(C1_PRESSURE, m1['pressure'])}")
print(f"{'gps_pos':>{ckfns.lpad}}: {ckfns.ck_err(C1_GPS_POS, m1['gps_pos'])}")
print(f"{'gps_vg_chi':>{ckfns.lpad}}: {_ck_err_either(C1_GPS_VG_CHI_A, C1_GPS_VG_CHI_B, m1['gps_vg_chi'])}")
print(f"{'gps_eta':>{ckfns.lpad}}: {ckfns.ck_err(C1_ETA, np.array([mav._gps_eta_n, mav._gps_eta_e, mav._gps_eta_h]))}")
print(f"{'t_gps':>{ckfns.lpad}}: {ckfns.ck_err(C1_T_GPS, mav._t_gps)}\n")

# ----------------------
# Case 2: GPS hold branch
print(f"\t{tools.color.cyan('### Case 2: GPS hold ###')}\n")

mav._state = np.array([
    [55.0],
    [-35.0],
    [-104.0],
    [24.2],
    [-0.35],
    [2.1],
    [0.9745],
    [0.1122],
    [-0.0453],
    [0.1868],
    [-0.11],
    [0.07],
    [-0.02],
])
mav._forces = np.array([[6.0], [-2.2], [-48.0]])
mav._Va = 23.7
mav._wind = np.array([[-3.2], [1.4], [0.0]])
mav._t_gps = 0.0

with _noise_as_mean():
    meas_case2 = mav.sensors()
m2 = _meas_groups(meas_case2)

print(f"{'gyro':>{ckfns.lpad}}: {ckfns.ck_err(C2_GYRO, m2['gyro'])}")
print(f"{'accel':>{ckfns.lpad}}: {ckfns.ck_err(C2_ACCEL, m2['accel'])}")
print(f"{'mag':>{ckfns.lpad}}: {_ck_err_either(C2_MAG, np.zeros(3), m2['mag'])}")
print(f"{'pressure':>{ckfns.lpad}}: {ckfns.ck_err(C2_PRESSURE, m2['pressure'])}")
print(f"{'gps_pos':>{ckfns.lpad}}: {ckfns.ck_err(C2_GPS_POS, m2['gps_pos'])}")
print(f"{'gps_vg_chi':>{ckfns.lpad}}: {_ck_err_either(C2_GPS_VG_CHI_A, C2_GPS_VG_CHI_B, m2['gps_vg_chi'])}")
print(f"{'gps_eta':>{ckfns.lpad}}: {ckfns.ck_err(C2_ETA, np.array([mav._gps_eta_n, mav._gps_eta_e, mav._gps_eta_h]))}")
print(f"{'t_gps':>{ckfns.lpad}}: {ckfns.ck_err(C2_T_GPS, mav._t_gps)}\n")
