import numpy as np
from models import model_coef as TF
import parameters.aerosonde_parameters as MAV


#### TODO #####
gravity = MAV.gravity  # gravity constant
Va0 = TF.Va_trim
rho = 1.0 # density of air
sigma = 0  # low pass filter gain for derivative

#----------roll loop-------------
# get transfer function data for delta_a to phi
wn_roll = 10.0
zeta_roll = .707
roll_kp = wn_roll**2 / TF.a_phi2
roll_kd = (2*zeta_roll*wn_roll-TF.a_phi1) / TF.a_phi2

#----------course loop-------------
wn_course = wn_roll / 20
zeta_course = 1.0
course_kp = 2*zeta_course*wn_course*Va0/gravity
course_ki = wn_course**2*Va0/gravity

#----------yaw damper-------------
yaw_damper_p_wo = .45
yaw_damper_kr = 2.0

#----------pitch loop-------------
wn_pitch = 15
zeta_pitch = .707
pitch_kp = (wn_pitch**2-TF.a_theta2) / TF.a_theta3
pitch_kd = (2*zeta_pitch*wn_pitch - TF.a_theta1) / TF.a_theta3
K_theta_DC = (pitch_kp*TF.a_theta3)/(TF.a_theta2 + pitch_kp*TF.a_theta3)

#----------altitude loop-------------
wn_altitude = wn_pitch / 30 # this is our Wh
zeta_altitude = 1.0
altitude_kp = 2*zeta_altitude*wn_altitude / K_theta_DC/Va0
altitude_ki = wn_altitude**2 / K_theta_DC/Va0
altitude_zone = 10.0

#---------airspeed hold using throttle---------------
wn_airspeed_throttle = 1.5
zeta_airspeed_throttle = 2.0
airspeed_throttle_kp = (2*zeta_airspeed_throttle*wn_airspeed_throttle - TF.a_V1) / TF.a_V2
airspeed_throttle_ki = wn_airspeed_throttle**2/TF.a_V2

print("lateral: ")
print("  roll_kp: ", roll_kp, ", roll_kd: ", roll_kd)
print("  course_kp: ", course_kp, ", course_ki: ", course_ki)
# print("  yaw_damper_kr: ", yaw_damper_kr, ", yaw_damper_p_wo: ", yaw_damper_p_wo)
print("longitudinal: ")
print("  pitch_kp: ", pitch_kp, ", pitch_kd: ", pitch_kd)
print("  altitude_kp: ", altitude_kp, ", altitude_ki: ", altitude_ki)
print("  airspeed_throttle_kp: ", airspeed_throttle_kp, ", airspeed_throttle_ki: ", airspeed_throttle_ki)