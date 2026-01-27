"""
various tools to be used in mavPySim
"""

# typing
import numpy.typing as npt

# 3rd party
import numpy as np

def quaternion_to_euler(quaternion: npt.NDArray[np.float64]) -> tuple[float]:
    """
    Converts a quaternion attitude to an euler angle attitude

    Args:
        quaternion: `size == 4` quaternion vector

    Returns:
        tuple of euler angles
    """
    e0 = quaternion.item(0)
    e1 = quaternion.item(1)
    e2 = quaternion.item(2)
    e3 = quaternion.item(3)
    phi = np.arctan2(2.0 * (e0 * e1 + e2 * e3), e0**2.0 + e3**2.0 - e1**2.0 - e2**2.0)
    # theta = np.arcsin(2.0 * (e0 * e2 - e1 * e3))
    theta = -np.pi/2.0 + 2*np.arctan2(np.sqrt(1+2.0*(e0*e2-e1*e3)), np.sqrt(1-2.0*(e0*e2-e1*e3)))
    psi = np.arctan2(2.0 * (e0 * e3 + e1 * e2), e0**2.0 + e1**2.0 - e2**2.0 - e3**2.0)
    return phi, theta, psi

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

    # ----------------------------------------------
    # an attitude (rotation) quaternion needs to be verifiably unit length

    # quat_vec = np.array([e0, e1, e2, e3])
    # mag_quat_vec = np.linalg.norm(quat_vec)

    # if np.abs(1 - mag_quat_vec) <= 1e-14:
    #     return np.array([[e0],[e1],[e2],[e3]])
    # elif mag_quat_vec >= 1e-14:
    #     qnorm = quat_vec / mag_quat_vec
    #     return np.atleast_2d(qnorm).T
    # else:
    #     print("Error: zero quaternion vector!")
    #     raise Exception('quaternion magnitude is zero')
    # #

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

def quaternion_to_rotation(quaternion: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """
    converts a quaternion attitude to a 3-2-1 rotation matrix (R_b^i  **b2i = transpose of the book**)
    """
    e0 = quaternion.item(0)
    e1 = quaternion.item(1)
    e2 = quaternion.item(2)
    e3 = quaternion.item(3)

    R = np.array([[e1 ** 2.0 + e0 ** 2.0 - e2 ** 2.0 - e3 ** 2.0, 2.0 * (e1 * e2 - e3 * e0), 2.0 * (e1 * e3 + e2 * e0)],
                  [2.0 * (e1 * e2 + e3 * e0), e2 ** 2.0 + e0 ** 2.0 - e1 ** 2.0 - e3 ** 2.0, 2.0 * (e2 * e3 - e1 * e0)],
                  [2.0 * (e1 * e3 - e2 * e0), 2.0 * (e2 * e3 + e1 * e0), e3 ** 2.0 + e0 ** 2.0 - e1 ** 2.0 - e2 ** 2.0]])
    R = R/linalg.det(R)

    return R

def rotation_to_quaternion(R: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """
    converts a 3-2-1 R_b^i (**b2i = transpose of the book!**) rotation matrix to a unit quaternion
    """
    r11 = R[0][0]
    r12 = R[0][1]
    r13 = R[0][2]
    r21 = R[1][0]
    r22 = R[1][1]
    r23 = R[1][2]
    r31 = R[2][0]
    r32 = R[2][1]
    r33 = R[2][2]

    # can't the determinant be used to allow removal of all the if statements?
    tmp=r11+r22+r33
    if tmp>0:
        e0 = 0.5*np.sqrt(1+tmp)
    else:
        e0 = 0.5*np.sqrt(((r12-r21)**2+(r13-r31)**2+(r23-r32)**2)/(3-tmp))

    tmp=r11-r22-r33
    if tmp>0:
        e1 = 0.5*np.sqrt(1+tmp)
    else:
        e1 = 0.5*np.sqrt(((r12+r21)**2+(r13+r31)**2+(r23-r32)**2)/(3-tmp))

    tmp=-r11+r22-r33
    if tmp>0:
        e2 = 0.5*np.sqrt(1+tmp)
    else:
        e2 = 0.5*np.sqrt(((r12+r21)**2+(r13+r31)**2+(r23+r32)**2)/(3-tmp))

    tmp=-r11+-22+r33
    if tmp>0:
        e3 = 0.5*np.sqrt(1+tmp)
    else:
        e3 = 0.5*np.sqrt(((r12-r21)**2+(r13+r31)**2+(r23+r32)**2)/(3-tmp))

    return np.array([[e0], [e1], [e2], [e3]])

def rotation_to_euler(R_b2i: npt.NDArray[np.float64]) -> tuple[float]:
    """
    converts a 3-2-1 R_b^i (**b2i = transpose of the book!**) rotation matrix to euler angles
    """
    # alternate method
    # quat = rotation_to_quaternion(R)
    # phi, theta, psi = quaternion_to_euler(quat)

    # --------------------------

    # arctan2 is more stable than arcsin !!

    # equivalently poor (only uses one number as source):
    # thetas = np.arcsin(-rpy_i2b[2,0])
    # theta0 = np.arctan2(-rpy_i2b[2,0], np.sqrt(1-rpy_i2b[2,0]**2))

    # equivalently better behaved (uses two numbers as source):
    # theta2 = np.arctan2(-rpy_i2b[2,0], np.sqrt(rpy_i2b[0,0]**2 + rpy_i2b[1,0]**2))
    # theta1 = np.arctan2(-rpy_i2b[2,0], np.sqrt(rpy_i2b[2,1]**2 + rpy_i2b[2,2]**2))

    phi     = np.arctan2(R_b2i[2,1], R_b2i[2,2])
    theta   = np.arctan2(-R_b2i[2,0], np.sqrt(R_b2i[0,0]**2 + R_b2i[1,0]**2))
    psi     = np.arctan2(R_b2i[1,0], R_b2i[0,0])

    return phi, theta, psi

def hat(omega: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """
    vector to skew symmetric matrix associated with cross product
    """
    a = omega.item(0)
    b = omega.item(1)
    c = omega.item(2)

    omega_hat = np.array([[0, -c, b],
                          [c, 0, -a],
                          [-b, a, 0]])
    return omega_hat

