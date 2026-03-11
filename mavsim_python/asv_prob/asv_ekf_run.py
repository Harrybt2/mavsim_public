# asv_ekf_run.py
#
# Starter script for the ASV EKF problem (Problem 3).
# This mimics the structure used for the UFO EKF problem:
#   1) load data
#   2) create EKF observer object
#   3) iterate through the log calling update(...)
#   4) compute requested metrics and generate plots

import numpy as np
import matplotlib.pyplot as plt
from asv_observer import EkfStateObserver


def wrap_angle(angle_rad: float) -> float:
    """Wrap angle to [-pi, pi)."""
    return (angle_rad + np.pi) % (2.0 * np.pi) - np.pi


# ----------------------------
# Load the data
# Data columns:
#   (t, u, delta, p_n_gps, p_e_gps, Vg_gps, chi_gps)
# ----------------------------
asv_data = np.loadtxt(open("asv_prob\\asv_data.txt", "rb"), delimiter=",")
t = asv_data[:, 0]
u = asv_data[:, 1]
delta = asv_data[:, 2]
pn_gps = asv_data[:, 3]
pe_gps = asv_data[:, 4]
Vg_gps = asv_data[:, 5]
chi_gps = asv_data[:, 6]
M = t.shape[0]

# State: x = [p_n, p_e, V, chi, c_n, c_e]^T
xhat = np.zeros((6, M))

# Create EKF object (students implement EkfStateObserver)
asv_ekf = EkfStateObserver(
    pn0=pn_gps[0],
    pe0=pe_gps[0],
    V0=Vg_gps[0],
    chi0=chi_gps[0],
)

for k in range(M):
    # Pack inputs for the EKF update.
    # Keep this ordering consistent with asv_observer.py
    inp = np.array([
        t[k],
        u[k],
        delta[k],
        pn_gps[k],
        pe_gps[k],
        Vg_gps[k],
        chi_gps[k],
    ])
    xhat[:, k:k+1] = asv_ekf.update(inp)

pn_hat = xhat[0, :]
pe_hat = xhat[1, :]
V_hat = xhat[2, :]
chi_hat = xhat[3, :]
cn_hat = xhat[4, :]
ce_hat = xhat[5, :]

# ----------------------------
# Metrics requested by the exam problem
# ----------------------------
print("Estimated current at end of run:")
print("  c_n(t_f) =", float(cn_hat[-1]), "m/s")
print("  c_e(t_f) =", float(ce_hat[-1]), "m/s")

max_speed = float(np.max(V_hat))
print("Maximum estimated speed =", max_speed, "m/s")

print("Mean planar position error metric not computed in this starter script.")

# ----------------------------
# Plots requested by the exam problem
# ----------------------------
# (1) pn and pe vs time
plt.figure(1)
plt.subplot(211)
plt.plot(t, pn_gps, label="p_n,gps")
plt.plot(t, pn_hat, label="p_n,hat")
plt.ylabel("north position (m)")
plt.grid(True)
plt.legend()

plt.subplot(212)
plt.plot(t, pe_gps, label="p_e,gps")
plt.plot(t, pe_hat, label="p_e,hat")
plt.xlabel("time (s)")
plt.ylabel("east position (m)")
plt.grid(True)
plt.legend()

# (2) speed vs time
plt.figure(2)
plt.plot(t, Vg_gps, label="V_g,gps")
plt.plot(t, V_hat, label="V,hat")
plt.xlabel("time (s)")
plt.ylabel("speed (m/s)")
plt.grid(True)
plt.legend()

# (3) current vs time
plt.figure(3)
plt.subplot(211)
plt.plot(t, cn_hat)
plt.ylabel("c_n hat (m/s)")
plt.grid(True)
plt.subplot(212)
plt.plot(t, ce_hat)
plt.xlabel("time (s)")
plt.ylabel("c_e hat (m/s)")
plt.grid(True)

# (4) innovations at GPS update times (optional)
# NOTE: to make this plot, store innovations inside EkfStateObserver and expose them.
# If you implement innovation logging, uncomment below.
#
# innov = asv_ekf.innovations  # expected shape: (4, num_updates)
# t_upd  = asv_ekf.innovation_times
# plt.figure(4)
# plt.subplot(211)
# plt.stem(t_upd, innov[0, :], use_line_collection=True)
# plt.ylabel("innov p_n")
# plt.grid(True)
# plt.subplot(212)
# plt.stem(t_upd, innov[1, :], use_line_collection=True)
# plt.xlabel("time (s)")
# plt.ylabel("innov p_e")
# plt.grid(True)

plt.show()
