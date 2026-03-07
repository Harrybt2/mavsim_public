"""
compute_ss_model
    - Chapter 5 assignment for Beard & McLain, PUP, 2012
    - Update history:  
        2/4/2019 - RWB
"""
import numpy as np
from scipy.optimize import minimize
from tools.rotations import euler_to_quaternion, quaternion_to_euler
import parameters.aerosonde_parameters as MAV
from parameters.simulation_parameters import ts_simulation as Ts
from message_types.msg_delta import MsgDelta


def compute_model(mav, trim_state, trim_input):
    # Note: this function alters the mav private variables
    A_lon, B_lon, A_lat, B_lat = compute_ss_model(mav, trim_state, trim_input)
    Va_trim, alpha_trim, theta_trim, a_phi1, a_phi2, a_theta1, a_theta2, a_theta3, \
    a_V1, a_V2, a_V3 = compute_tf_model(mav, trim_state, trim_input)

    # write transfer function gains to file
    file = open('models/model_coef.py', 'w')
    file.write('import numpy as np\n')
    file.write('x_trim = np.array([[%f, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f]]).T\n' %
               (trim_state.item(0), trim_state.item(1), trim_state.item(2), trim_state.item(3),
                trim_state.item(4), trim_state.item(5), trim_state.item(6), trim_state.item(7),
                trim_state.item(8), trim_state.item(9), trim_state.item(10), trim_state.item(11),
                trim_state.item(12)))
    file.write('u_trim = np.array([[%f, %f, %f, %f]]).T\n' %
               (trim_input.elevator, trim_input.aileron, trim_input.rudder, trim_input.throttle))
    file.write('Va_trim = %f\n' % Va_trim)
    file.write('alpha_trim = %f\n' % alpha_trim)
    file.write('theta_trim = %f\n' % theta_trim)
    file.write('a_phi1 = %f\n' % a_phi1)
    file.write('a_phi2 = %f\n' % a_phi2)
    file.write('a_theta1 = %f\n' % a_theta1)
    file.write('a_theta2 = %f\n' % a_theta2)
    file.write('a_theta3 = %f\n' % a_theta3)
    file.write('a_V1 = %f\n' % a_V1)
    file.write('a_V2 = %f\n' % a_V2)
    file.write('a_V3 = %f\n' % a_V3)
    file.write('A_lon = np.array([\n    [%f, %f, %f, %f, %f],\n    '
               '[%f, %f, %f, %f, %f],\n    '
               '[%f, %f, %f, %f, %f],\n    '
               '[%f, %f, %f, %f, %f],\n    '
               '[%f, %f, %f, %f, %f]])\n' %
    (A_lon[0][0], A_lon[0][1], A_lon[0][2], A_lon[0][3], A_lon[0][4],
     A_lon[1][0], A_lon[1][1], A_lon[1][2], A_lon[1][3], A_lon[1][4],
     A_lon[2][0], A_lon[2][1], A_lon[2][2], A_lon[2][3], A_lon[2][4],
     A_lon[3][0], A_lon[3][1], A_lon[3][2], A_lon[3][3], A_lon[3][4],
     A_lon[4][0], A_lon[4][1], A_lon[4][2], A_lon[4][3], A_lon[4][4]))
    file.write('B_lon = np.array([\n    [%f, %f],\n    '
               '[%f, %f],\n    '
               '[%f, %f],\n    '
               '[%f, %f],\n    '
               '[%f, %f]])\n' %
    (B_lon[0][0], B_lon[0][1],
     B_lon[1][0], B_lon[1][1],
     B_lon[2][0], B_lon[2][1],
     B_lon[3][0], B_lon[3][1],
     B_lon[4][0], B_lon[4][1],))
    file.write('A_lat = np.array([\n    [%f, %f, %f, %f, %f],\n    '
               '[%f, %f, %f, %f, %f],\n    '
               '[%f, %f, %f, %f, %f],\n    '
               '[%f, %f, %f, %f, %f],\n    '
               '[%f, %f, %f, %f, %f]])\n' %
    (A_lat[0][0], A_lat[0][1], A_lat[0][2], A_lat[0][3], A_lat[0][4],
     A_lat[1][0], A_lat[1][1], A_lat[1][2], A_lat[1][3], A_lat[1][4],
     A_lat[2][0], A_lat[2][1], A_lat[2][2], A_lat[2][3], A_lat[2][4],
     A_lat[3][0], A_lat[3][1], A_lat[3][2], A_lat[3][3], A_lat[3][4],
     A_lat[4][0], A_lat[4][1], A_lat[4][2], A_lat[4][3], A_lat[4][4]))
    file.write('B_lat = np.array([\n    [%f, %f],\n    '
               '[%f, %f],\n    '
               '[%f, %f],\n    '
               '[%f, %f],\n    '
               '[%f, %f]])\n' %
    (B_lat[0][0], B_lat[0][1],
     B_lat[1][0], B_lat[1][1],
     B_lat[2][0], B_lat[2][1],
     B_lat[3][0], B_lat[3][1],
     B_lat[4][0], B_lat[4][1],))
    file.write('Ts = %f\n' % Ts)
    file.close()


def compute_tf_model(mav, trim_state, trim_input):
    # trim values
    mav._state = trim_state
    mav._update_velocity_data()
    Va_trim = mav._Va
    alpha_trim = mav._alpha
    phi, theta_trim, psi = quaternion_to_euler(trim_state[6:10])

    ###### TODO ######
    # define transfer function constants
    # slide 21
    a_phi1 = -0.5*MAV.rho*mav._Va**2*MAV.S_wing*MAV.b*MAV.C_p_p*MAV.b/(2*mav._Va)
    a_phi2 = 0.5*MAV.rho*mav._Va**2*MAV.S_wing*MAV.b*MAV.C_p_delta_a
    #slide 28
    a_theta1 = -MAV.rho*mav._Va**2*MAV.c*MAV.S_wing/(2*MAV.Jy)*MAV.C_m_q*MAV.c/(2*mav._Va)
    a_theta2 = -MAV.rho*mav._Va**2*MAV.c*MAV.S_wing/(2*MAV.Jy)*MAV.C_m_alpha
    a_theta3 = MAV.rho*mav._Va**2*MAV.c*MAV.S_wing/(2*MAV.Jy)*MAV.C_m_delta_e

    # Compute transfer function coefficients using new propulsion model
    #slide 34
    a_V1 = MAV.rho*Va_trim*MAV.S_wing/MAV.mass*(MAV.C_D_0+MAV.C_D_alpha*alpha_trim+MAV.C_D_delta_e*trim_input.elevator)-(1/MAV.mass)*dT_dVa(mav,Va_trim,trim_input.throttle)
    a_V2 = 1/MAV.mass*dT_ddelta_t(mav,Va_trim,trim_input.throttle)
    a_V3 = MAV.gravity*np.cos(theta_trim-alpha_trim)

    return Va_trim, alpha_trim.item(), theta_trim.item(), a_phi1, a_phi2, a_theta1, a_theta2, a_theta3, a_V1.item(), a_V2.item(), a_V3.item()


def compute_ss_model(mav, trim_state, trim_input):
    x_euler = euler_state(trim_state)
    
    ##### TODO #####
    A = df_dx(mav, x_euler, trim_input)
    B = df_du(mav, x_euler, trim_input)
    # extract longitudinal states (u, w, q, theta, pd)
    A_lon = np.zeros((5,5))
    idx_a_lon = [3,5,10,7,2]
    A_lon = A[np.ix_(idx_a_lon, idx_a_lon)]
    # A_lon[0] = A[3][:5] # place u row
    # A_lon[1] = A[5][:5] # place w row
    # A_lon[2] = A[10][:5] # place q row 
    # A_lon[3] = A[7][:5] # place theta row 
    # A_lon[4] = A[2][:5] # place pd row 

    A_lon[:,-1]*=-1
    A_lon[-1,:]*=-1

    B_lon = np.zeros((5,2))
    idx_b_lon = [0,3]
    B_lon = B[np.ix_(idx_a_lon, idx_b_lon)]

    B_lon[-1]*=-1
    # B_lon[0] = B[3][:2] # place u row
    # B_lon[1] = B[5][:2] # place w row
    # B_lon[2] = B[10][:2] # place q row 
    # B_lon[3] = B[7][:2] # place theta row 
    # B_lon[4] = B[2][:2] # place pd row 

    # change pd to h

    # extract lateral states (v, p, r, phi, psi)
    A_lat = np.zeros((5,5))
    idx_a_lat = [4,9,11,6,8]
    A_lat = A[np.ix_(idx_a_lat, idx_a_lat)]
    # A_lat[0] = A[4][5:10] # place v row
    # A_lat[1] = A[9][5:10] # place p row
    # A_lat[2] = A[11][5:10] # place r row 
    # A_lat[3] = A[6][5:10] # place phi row 
    # A_lat[4] = A[8][5:10] # place psi row     

    B_lat = np.zeros((5,2))
    id_x_b_lat = [1,2]
    B_lat = B[np.ix_(idx_a_lat, id_x_b_lat)]
    # B_lat[0] = B[4][2:] # place v row
    # B_lat[1] = B[9][2:] # place p row
    # B_lat[2] = B[11][2:] # place r row 
    # B_lat[3] = B[6][2:] # place phi row 
    # B_lat[4] = B[8][2:] # place psi row  

    return A_lon, B_lon, A_lat, B_lat

def euler_state(x_quat):
    # convert state x with attitude represented by quaternion
    # to x_euler with attitude represented by Euler angles
    
    ##### TODO #####
    e0 = x_quat.item(6)
    e1 = x_quat.item(7)
    e2 = x_quat.item(8)
    e3 = x_quat.item(9)
    quaternion = np.array([e0,e1,e2,e3])
    phi,theta,psi = quaternion_to_euler(quaternion)
    # x_euler = np.zeros((12,1))
    x_euler = np.copy(x_quat)
    x_euler[6]=phi
    x_euler[7]=theta
    x_euler[8]=psi
    x_euler = np.delete(x_euler,9)
    x_euler = x_euler.reshape((12,1))
    return x_euler

def quaternion_state(x_euler):
    # convert state x_euler with attitude represented by Euler angles
    # to x_quat with attitude represented by quaternions
    es = euler_to_quaternion(x_euler[6], x_euler[7], x_euler[8])
    
    ##### TODO #####
    x_quat = np.zeros((13,1))
    x_quat[0:6] = x_euler[0:6]
    x_quat[6] = es[0]
    x_quat[7] = es[1]
    x_quat[8] = es[2]
    x_quat[9] = es[3]
    x_quat[10:13] = x_euler[9:12]

    return x_quat

def f_euler(mav, x_euler, delta):
    # return 12x1 dynamics (as if state were Euler state)
    # compute f at euler_state, f_euler will be f, except for the attitude states

    # need to correct attitude states by multiplying f by
    # partial of quaternion_to_euler(quat) with respect to quat
    # compute partial quaternion_to_euler(quat) with respect to quat
    # dEuler/dt = dEuler/dquat * dquat/dt

    x_quat = quaternion_state(x_euler)
    mav._state = x_quat
    mav._update_velocity_data()
    
    # Get forces and moments, then compute state derivatives in quaternion form
    forces_moments = mav._forces_moments(delta)
    f_quat = mav._f(x_quat, forces_moments)
    dquat_dt = f_quat[6:10]
    f_euler_ =euler_state(f_quat)

    # run quaternion to euler pertubring e0,e1,e2,e3  and divide, populate a 3x4 matrix 
    # that 3x4 multiplies dquat_dt populate that into Feuler[6:9]

    eps = 0.001
    e = x_quat[6:10]
    euler_angs= x_euler[6:9]
    phi = x_euler.item(6)
    theta = x_euler.item(7)
    psi = x_euler.item(8)

    deuler_dt = np.zeros((len(euler_angs), len(dquat_dt)))
    dTheta_dquat = np.zeros((3,4))
    for i in range(len(dquat_dt)):
        tmp = np.zeros((4,1))
        tmp[i][0] = eps
        e_eps = (e+tmp)/np.linalg.norm(e+tmp)
        phi_eps, theta_eps, psi_eps = quaternion_to_euler(e_eps)

        dTheta_dquat[0][i] = (phi_eps - phi) / eps
        dTheta_dquat[1][i] = (theta_eps - theta) / eps
        dTheta_dquat[2][i] = (psi_eps - psi) / eps
        # dquat_dt_eps = np.copy(dquat_dt) # copy all the e's
        # dquat_dt_eps[i] += eps # perturb one of the e's
        # euler_eps= quaternion_to_euler(dquat_dt_eps.flatten()) # find all the euler angles for the perturbed e
        # for j in range(len(euler_angs)): # for each of the euler angles, find the partial derivative with respect to the perturbed e
        #     deuler_dt[j][i] = (euler_angs[j] - euler_eps[j]) / eps

    f_euler_ = np.zeros((12, 1))

    f_euler_[:6] = f_quat[:6]
    f_euler_[6:9] = np.copy(dTheta_dquat @ f_quat[6:10])
    f_euler_[9:] = f_quat[10:]
    # f_euler_[6:9] = f_euler_[6:9] * deuler_dt.reshape((3,1))

    # dEuler_dQuat = deuler_dt.reshape((3,1)) / dquat_dt.reshape((4,1))
    
    
    
    # # Compute Jacobian of quaternion_to_euler with respect to quaternion (3x4 matrix)
    # eps = 0.001
    # quaternion = x_quat[6:10].flatten()
    # euler_0 = np.array(quaternion_to_euler(quaternion)).reshape((3, 1))
    
    # dEuler_dquat = np.zeros((3, 4))
    # for i in range(4):
    #     quat_eps = np.copy(quaternion)
    #     quat_eps[i] += eps
    #     euler_eps = np.array(quaternion_to_euler(quat_eps)).reshape((3, 1))
    #     dEuler_dquat[:, i] = ((euler_eps - euler_0) / eps).flatten()
    
    # # Apply chain rule: dEuler/dt = dEuler/dquat @ dquat/dt
    
    # f_euler_[:6] = f_quat[:6]  # position and velocity derivatives unchanged
    # f_euler_[6:9] = dEuler_dquat @ f_quat[6:10]  # attitude derivatives via chain rule
    # f_euler_[9:12] = f_quat[10:13]  # angular rate derivatives

    return f_euler_

# def df_dx(mav, x_euler, delta):
#     # take partial of f_euler with respect to x_euler
#     eps = 0.01  # deviation
#     f_euler0 = f_euler(mav, x_euler, delta)
#     f_euler1 = f_euler(mav, x_euler + eps, delta)
#     ##### TODO #####
#     A = np.zeros((12, 12))  # Jacobian of f wrt x
#     for i in range(12):
#         A[:,i] = (f_euler1[:,0] - f_euler0[:,0]) / eps
#     return A

def df_dx(mav, x_euler, delta):
    # take partial of f(x_euler, delta) with respect to x_euler
    eps = 0.01  # deviation
    A = np.zeros((12, 12))  # Jacobian of f wrt x
    f_at_x = f_euler(mav, x_euler, delta)
    for i in range(len(x_euler)):
        x_euler_eps = x_euler.copy()
        x_euler_eps[i] += eps
        f_at_x_eps = f_euler(mav, x_euler_eps, delta )
        A[:,i] = ((f_at_x_eps - f_at_x) / eps).flatten()

    return A


def df_du(mav, x_euler, delta):
    # take partial of f_euler with respect to input
    B = np.zeros((12, 4))  # Jacobian of f wrt u
    eps = 0.01  # deviation
    f_euler0 = f_euler(mav, x_euler, delta)
    for i in range(4):
        delta_eps = MsgDelta(elevator=delta.elevator, aileron=delta.aileron, rudder=delta.rudder, throttle=delta.throttle)
        if i == 0:
            delta_eps.elevator += eps
        elif i == 1:
            delta_eps.aileron += eps
        elif i == 2:
            delta_eps.rudder += eps
        elif i == 3:
            delta_eps.throttle += eps
        f_euler1 = f_euler(mav, x_euler, delta_eps)
        B[:,i] = ((f_euler1- f_euler0) / eps).flatten()
    ##### TODO #####
    
    
    # for i in range(4):
    #     B[:,i] = (f_euler1[:,0] - f_euler0[:,0]) / eps
    return B


def dT_dVa(mav, Va, delta_t):
    # returns the derivative of motor thrust with respect to Va
    eps = 0.01

    ##### TODO #####
    # Evaluate thrust at nominal throttle
    thrust_0, _ = mav._motor_thrust_torque(Va, delta_t)
    
    # Evaluate thrust at perturbed throttle
    thrust_eps, _ = mav._motor_thrust_torque(Va+eps, delta_t)
    
    # Finite difference approximation
    dT_dVa = (thrust_eps - thrust_0) / eps
    return dT_dVa


    
def dT_ddelta_t(mav, Va, delta_t):
    # returns the derivative of motor thrust with respect to delta_t
    eps = 0.01
    ##### TODO #####
    # Evaluate thrust at nominal throttle
    thrust_0, _ = mav._motor_thrust_torque(Va, delta_t)
    
    # Evaluate thrust at perturbed throttle
    thrust_eps, _ = mav._motor_thrust_torque(Va, delta_t + eps)
    
    # Finite difference approximation
    dT_ddelta_t = (thrust_eps - thrust_0) / eps
    
    return dT_ddelta_t

