# dubins_parameters
#   - Dubins parameters that define path between two configurations
#
# mavsim_matlab 
#     - Beard & McLain, PUP, 2012
#     - Update history:  
#         3/26/2019 - RWB
#         4/2/2020 - RWB
#         3/30/2022 - RWB
#         7/13/2023 - RWB
#         3/27/2024 - RWB
import numpy as np
from numpy import pi as pi

class DubinsParameters:
    '''
    Class that contains parameters for a Dubin's car path

    Attributes
    ----------
        p_s : np.ndarray (3x1)
            inertial position of start position, in meters
        chi_s : float
            course of start position in radians, measured from North
        p_e : np.ndarray (3x1)
            inertial position of end position, in meters
        chi_e : float
            course of end position in radians, measured from North
        R : float
            radius of start and end circles, from north
        center_s : np.ndarray (3x1)
            inertial center of start circle
        dir_s : int 
            direction of start circle: +1 CW, -1 CCW
        center_e : np.ndarray (3x1)
            inertial center of end circle
        dir_e : int 
            direction of end circle: +1 CW, -1 CCW
        length : float
            length of straight line segment
        r1 : np.ndarray (3x1)
            position on half plane for transition from start circle to straight-line
        n1 : np.ndarray (3x1)
            unit vector defining half plane for transition from start circle to straight-line, and from straight line to end circle
        r2 : np.ndarray (3x1)
            position on half plane for transition from straight line to end circle
        r3 : np.ndarray (3x1)
            position on half plane for end of dubins path
        n3 : np.ndarray (3x1)
            unit vector defining half plane for end of dubins path

    Methods
    ----------
    update(ps, chis, pe, chie, R)
        : create new Dubins path from start to end poses, with specified radius
    compute_parameters()
        : construct four dubins paths and pick the shortest and define all associated parameters.
    compute_points()
        : find equally spaced points along dubins path - for plotting and collision checking
    '''

    def update(self, 
               ps: np.ndarray, # (3x1) 
               chis: float, 
               pe: np.ndarray, # (3x1)
               chie: float, 
               R: float):
         self.p_s = ps
         self.chi_s = chis
         self.p_e = pe
         self.chi_e = chie
         self.radius = R
         
         self.compute_parameters()

    def compute_parameters(self):
        ps = self.p_s.reshape(3,1)
        pe = self.p_e.reshape(3,1)
        chis = self.chi_s
        chie = self.chi_e
        R = self.radius
        ell = np.linalg.norm(ps[0:2] - pe[0:2])
        e1 = np.array([[1, 0, 0]]).T  # unit vector along x-axis (north)

        ##### TODO #####
        if ell < 2 * R:
            print('Error in Dubins Parameters: The distance between nodes must be larger than 2R.')
            # raise Exception
        else:
            # compute start and end circles
            crs = ps + R*rotz(pi/2) @ np.array([[np.cos(chis)],[np.sin(chis)],[0]])
            cls = ps + R*rotz(-pi/2) @ np.array([[np.cos(chis)],[np.sin(chis)],[0]])
            cre = pe + R*rotz(pi/2) @ np.array([[np.cos(chie)],[np.sin(chie)],[0]])
            cle = pe + R*rotz(-pi/2) @ np.array([[np.cos(chie)],[np.sin(chie)],[0]])
            # TODO ell in the book (above eq 11.11 is slightly different than the ell provided above)

            # compute L1 eq 11.9 R-S-R
            # TODO maybe these are off? I really don't know
            theta = np.arctan2(
                    (cre - crs).item(1),
                    (cre - crs).item(0))
            theta = mod(theta) # TODO do you need to mod it here or just in the eqaution for L?

            L1 = np.linalg.norm(crs - cre) + R * mod(2*pi + mod(theta-pi/2)-mod(chis-pi/2)) + R * mod(2*pi + mod(chie-pi/2)-mod(theta-pi/2))

            # compute L2 eq 11.10 R-S-L
            theta = np.arctan2(
                    (cle- crs).item(1),
                    (cle- crs).item(0))
            theta = mod(theta)
            ell = np.linalg.norm(cle-crs)
            theta2 = theta - pi/2 + np.asin(2*R/ell)
            if np.isnan(theta2):
                L2 = np.inf
            else:
                theta2 = mod(theta2)
                L2 = np.sqrt(ell**2-4*R**2) + R*mod(2*pi + mod(theta2) - mod(chis - pi/2)) + R*mod(2*pi + mod(theta2 + pi) - mod(chie + pi/2))

            # compute L3 eq 11.11 L-S-R
            theta = np.arctan2(
                    (cre - cls).item(1),
                    (cre- cls).item(0))
            theta = mod(theta)
            ell = np.linalg.norm(cre-cls)
            theta2 = np.acos(2*R/ell)
            if np.isnan(theta2):
                L3 = np.inf
            else:
                theta2 = mod(theta2)
                L3 = np.sqrt(ell**2 - 4*R**2) + R*mod(2*pi + mod(chis + pi/2)-mod(theta+theta2)) + R*mod(2*pi + mod(chie - pi/2)-mod(theta + theta2 - pi))

            # compute L4 eq11.12 L-S-L
            theta = np.arctan2(
                    (cle- cls).item(1),
                    (cle- cls).item(0))
            theta = mod(theta)
            L4 = np.linalg.norm(cls - cle) + R*mod(2*pi + mod(chis + pi/2)-mod(theta + pi/2)) + R*mod(2*pi + mod(theta + pi/2)-mod(chie + pi/2))

            # L is the minimum distance
            
            
            L = np.min([L1, L2, L3, L4])
            min_idx = int(np.argmin([L1, L2, L3, L4]))
            

            # Debug prints
            # print(f"L1 (RSR): {L1}")
            # print(f"L2 (RSL): {L2}")
            # print(f"L3 (LSR): {L3}")
            # print(f"L4 (LSL): {L4}")
            # print(f"Selected case: {min_idx+1} (0=RSR, 1=RSL, 2=LSR, 3=LSL)")

            if min_idx == 0:
                self.center_s = crs
                self.dir_s = 1
                self.center_e = cre
                self.dir_e = 1
                self.n1 = (self.center_e-self.center_s)/np.linalg.norm(self.center_e-self.center_s)
                self.r1 = self.center_s + R*rotz(-pi/2)@self.n1
                self.r2 = self.center_e + R*rotz(-pi/2)@self.n1

            elif min_idx == 1:
                self.center_s = crs
                self.dir_s = 1
                self.center_e = cle
                self.dir_e = -1
                ell = np.linalg.norm(self.center_e-self.center_s)
                # TODO check if this angle is right
                theta = np.arctan2(
                    (self.center_e - self.center_s).item(1),
                    (self.center_e- self.center_s).item(0))
                theta2 = theta - pi/2 + np.asin(2*R/ell)
                self.n1 = rotz(theta2 + pi/2) @ e1
                self.r1 = self.center_s + R*rotz(theta2)@e1
                self.r2 = self.center_e + R*rotz(theta2 + pi)@e1

            elif min_idx == 2:
                self.center_s = cls
                self.dir_s = -1
                self.center_e = cre
                self.dir_e = 1
                ell = np.linalg.norm(self.center_e-self.center_s)
                # TODO check if this angle is right
                theta = np.arctan2(
                    (self.center_e - self.center_s).item(1),
                    (self.center_e- self.center_s).item(0))
                theta2 = np.acos(2*R/ell)
                self.n1 = rotz(theta + theta2 - pi/2) @ e1
                self.r1 = self.center_s + R*rotz(theta + theta2)@e1
                self.r2 = self.center_e + R*rotz(theta + theta2 - pi)@e1
            elif min_idx == 3:
                self.center_s = cls
                self.dir_s = -1
                self.center_e = cle
                self.dir_e = -1
                self.n1 = (self.center_e-self.center_s)/np.linalg.norm(self.center_e-self.center_s)
                self.r1 = self.center_s + R*rotz(pi/2)@self.n1
                self.r2 = self.center_e + R*rotz(pi/2)@self.n1
            
            self.length = L
            self.r3 = pe
            self.n3 = rotz(self.chi_e)@e1

    def compute_points(self):
        ##### TODO ##### - uncomment lines and remove last line
        Del = 0.1  # distance between point

        # points along start circle
        th1 = np.arctan2(self.p_s.item(1) - self.center_s.item(1),
                         self.p_s.item(0) - self.center_s.item(0))
        th1 = mod(th1)
        th2 = np.arctan2(self.r1.item(1) - self.center_s.item(1),
                         self.r1.item(0) - self.center_s.item(0))
        th2 = mod(th2)
        th = th1
        theta_list = [th]
        if self.dir_s > 0:
            if th1 >= th2:
                while th < th2 + 2*pi - Del:
                    th += Del
                    theta_list.append(th)
            else:
                while th < th2 - Del:
                    th += Del
                    theta_list.append(th)
        else:
            if th1 <= th2:
                while th > th2 - 2*pi + Del:
                    th -= Del
                    theta_list.append(th)
            else:
                while th > th2 + Del:
                    th -= Del
                    theta_list.append(th)

        points = np.array([[self.center_s.item(0) + self.radius * np.cos(theta_list[0]),
                            self.center_s.item(1) + self.radius * np.sin(theta_list[0]),
                            self.center_s.item(2)]])
        for angle in theta_list:
            new_point = np.array([[self.center_s.item(0) + self.radius * np.cos(angle),
                                   self.center_s.item(1) + self.radius * np.sin(angle),
                                   self.center_s.item(2)]])
            points = np.concatenate((points, new_point), axis=0)

        # points along straight line
        sig = 0
        while sig <= 1:
            new_point = np.array([[(1 - sig) * self.r1.item(0) + sig * self.r2.item(0),
                                   (1 - sig) * self.r1.item(1) + sig * self.r2.item(1),
                                   (1 - sig) * self.r1.item(2) + sig * self.r2.item(2)]])
            points = np.concatenate((points, new_point), axis=0)
            sig += Del

        # points along end circle
        th2 = np.arctan2(self.p_e.item(1) - self.center_e.item(1),
                         self.p_e.item(0) - self.center_e.item(0))
        th2 = mod(th2)
        th1 = np.arctan2(self.r2.item(1) - self.center_e.item(1),
                         self.r2.item(0) - self.center_e.item(0))
        th1 = mod(th1)
        th = th1
        theta_list = [th]
        if self.dir_e > 0:
            if th1 >= th2:
                while th < th2 + 2 * pi - Del:
                    th += Del
                    theta_list.append(th)
            else:
                while th < th2 - Del:
                    th += Del
                    theta_list.append(th)
        else:
            if th1 <= th2:
                while th > th2 - 2 * pi + Del:
                    th -= Del
                    theta_list.append(th)
            else:
                while th > th2 + Del:
                    th -= Del
                    theta_list.append(th)
        for angle in theta_list:
            new_point = np.array([[self.center_e.item(0) + self.radius * np.cos(angle),
                                   self.center_e.item(1) + self.radius * np.sin(angle),
                                   self.center_e.item(2)]])
            points = np.concatenate((points, new_point), axis=0)
        # points = np.zeros((5,3))
        return points


def rotz(theta: float):
    '''
    returns rotation matrix for right handed passive rotation about z-axis
    '''
    return np.array([[np.cos(theta), -np.sin(theta), 0],
                    [np.sin(theta), np.cos(theta), 0],
                    [0, 0, 1]])


def mod(x: float):
    '''
    wrap x to be between 0 and 2*pi
    '''
    while x < 0:
        x += 2*pi
    while x > 2*pi:
        x -= 2*pi
    return x


