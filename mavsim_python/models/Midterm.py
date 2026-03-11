import numpy as np
import numpy.typing as npt
# from tools.rotations import quaternion_to_rotation, quaternion_to_euler, euler_to_rotation

def euler_to_quaternion(
    phi : float,
    theta : float,
    psi : float,
) -> npt.NDArray[np.float64]:
    """
    Converts an euler angle attitude to a quaternian attitude

    Args:
        param euler: Euler angle attitude in three floats, phi, theta, psi

    Returns:
        Quaternian attitude in 2d np.array
    """

    e0 = np.cos(psi/2.0) * np.cos(theta/2.0) * np.cos(phi/2.0) + np.sin(psi/2.0) * np.sin(theta/2.0) * np.sin(phi/2.0)
    e1 = np.cos(psi/2.0) * np.cos(theta/2.0) * np.sin(phi/2.0) - np.sin(psi/2.0) * np.sin(theta/2.0) * np.cos(phi/2.0)
    e2 = np.cos(psi/2.0) * np.sin(theta/2.0) * np.cos(phi/2.0) + np.sin(psi/2.0) * np.cos(theta/2.0) * np.sin(phi/2.0)
    e3 = np.sin(psi/2.0) * np.cos(theta/2.0) * np.cos(phi/2.0) - np.cos(psi/2.0) * np.sin(theta/2.0) * np.sin(phi/2.0)

    return np.array([[e0],[e1],[e2],[e3]])

def euler_to_rotation(
    phi: float,
    theta: float,
    psi: float,
) -> npt.NDArray[np.float64]:
    """
    Converts euler angles to 3-2-1 rotation matrix (R_b^i  **b2i = transpose of the book**)
    """
    c_phi = np.cos(phi)
    s_phi = np.sin(phi)
    c_theta = np.cos(theta)
    s_theta = np.sin(theta)
    c_psi = np.cos(psi)
    s_psi = np.sin(psi)

    R_roll = np.array([[1, 0, 0],
                       [0, c_phi, -s_phi],
                       [0, s_phi, c_phi]])
    R_pitch = np.array([[c_theta, 0, s_theta],
                        [0, 1, 0],
                        [-s_theta, 0, c_theta]])
    R_yaw = np.array([[c_psi, -s_psi, 0],
                      [s_psi, c_psi, 0],
                      [0, 0, 1]])
    #R = np.dot(R_yaw, np.dot(R_pitch, R_roll))
    R = R_yaw @ R_pitch @ R_roll

    # rotation is body to inertial frame
    # R = np.array([[c_theta*c_psi, s_phi*s_theta*c_psi-c_phi*s_psi, c_phi*s_theta*c_psi+s_phi*s_psi],
    #               [c_theta*s_psi, s_phi*s_theta*s_psi+c_phi*c_psi, c_phi*s_theta*s_psi-s_phi*c_psi],
    #               [-s_theta, s_phi*c_theta, c_phi*c_theta]])

    return R


# %% prblm 2
phi = np.deg2rad(-15)
theta = np.deg2rad(4)
psi = np.deg2rad(75)

es = euler_to_quaternion(phi,theta,psi)
print(es)

R_body_to_inertial = euler_to_rotation(phi,theta,psi)
vel_body = np.array([[42],[3],[-6]])
print(R_body_to_inertial@vel_body)


# %% prblm 4
phi = np.deg2rad(12)
theta = np.deg2rad(-3)
psi = np.deg2rad(130) 

V_g_in_b = np.array([[28],[-1.5],[2.5]])
V_wind_in_inertial = np.array([[-3],[4],[0]])

R_body_to_inertial = euler_to_rotation(phi,theta,psi)

V_a_in_body = R_body_to_inertial.T @ V_wind_in_inertial
print("V_a_in_body: ", V_a_in_body)
print("total Va:, \n", np.linalg.norm(V_a_in_body))

u_r = V_a_in_body[0]
v_r = V_a_in_body[1]
w_r = V_a_in_body[2]

alpha = np.atan2(w_r,u_r)
print("angle of attack: ", np.rad2deg(alpha))

beta = np.atan2(v_r, np.sqrt(u_r**2 +w_r**2))
print("sideslip angle beta: ", np.rad2deg(beta))

gamma_a = theta-alpha
print("airmass referenced flight path angle gamma_a: ", np.rad2deg(gamma_a))

# %% prblm 5
g = 9.81
A = np.zeros((4,4))
B = np.zeros((4,2))
chi_star = np.deg2rad(30)
Vg_star = 25
phi_star = 0
a_star = 0
A[0][2] = np.cos(chi_star)
A[1][2] = np.sin(chi_star)
A[3][2] = -g/Vg_star**2 * np.tan(phi_star)
A[0][3] = -Vg_star*np.sin(chi_star)
A[1][3] = Vg_star*np.cos(chi_star)

print("A matrix evaluated at trim: \n", A)

B_entry = (g/Vg_star)*(1/np.cos(phi_star))**2
print('B\'s other entry that\'s not a 1: \n', B_entry )
B[2][0] = 1
B[3][1] = B_entry
Ts = 0.1
Ad = 0

Ad = (A*Ts)**0/(1) +(A*Ts)**1/(1) +(A*Ts)**2/(2)+ (A*Ts)**3/(6)
Bd = (np.ones(4)-Ad) @ B
print("Ad \n", Ad)
print("Bd \n", Bd)

x0 = np.array([[0],[0],[1.5],[np.deg2rad(-4)]])
u0 = np.array([[0.2],[np.deg2rad(10)]])

x1 = Ad @x0 + Bd @ u0

print("x1: \n", x1)