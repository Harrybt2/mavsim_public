import numpy as np

def wrap_angle(angle_rad: float) -> float:
    """Wrap angle to [-pi, pi)."""
    return (angle_rad + np.pi) % (2.0 * np.pi) - np.pi


class EkfStateObserver:
    """Starter framework for ASV EKF (Problem 3).

    State:
        x = [p_n, p_e, V, chi, c_n, c_e]^T
    Input:
        u_in = [u, delta]^T
    Measurement:
        y = [p_n_gps, p_e_gps, Vg_gps, chi_gps]^T
    """

    def __init__(self, pn0=0.0, pe0=0.0, V0=0.0, chi0=0.0):
        # TODO: Populate all model parameters from the exam statement.
        self.aV = 0.4 # s^-1
        self.bV = 2.0 # m/s^2
        self.kchi = 0.55 # s^-1
        self.Ts = .05 # s

        # TODO: Populate process/measurement noise covariances from exam statement.
        self.Q = np.diag([0.2**2, 0.2**2, 0.6**2, np.deg2rad(2)**2, .05**2, .05**2])
        self.R = np.diag([1.5**2, 1.5**2, .25**2, np.deg2rad(3)**2])

        # Initial state
        self.xhat = np.array([[pn0], [pe0], [V0], [chi0], [0.0], [0.0]], dtype=float)

        # TODO: Populate initial covariance P0 from exam statement.
        self.P = np.diag([10**2, 10**2, 2**2, np.deg2rad(10)**2, 1**2, 1**2])

    def f(self, x, u_in):
        """Continuous-time model xdot = f(x,u).

        TODO:
            Implement the dynamics from Problem 3.
        """
        p_n = x.item(0)
        p_e = x.item(1)
        V = x.item(2)
        chi = x.item(3)
        c_n = x.item(4)
        c_e = x.item(5)

        u = u_in.item(0)
        delta = u_in.item(1)

        # initialize what I'll return
        x_dot = np.zeros((6,1))

        p_n_dot = V*np.cos(chi) + c_n
        p_e_dot = V*np.sin(chi) + c_e
        V_dot = -self.aV*V + self.bV*u
        chi_dot = self.kchi*delta
        c_n_dot = 0
        c_e_dot = 0

        x_dot[0]=p_n_dot
        x_dot[1]=p_e_dot
        x_dot[2]=V_dot
        x_dot[3]=chi_dot
        x_dot[4]=c_n_dot
        x_dot[5]=c_e_dot

        # Placeholder so the template runs before completion.
        return x_dot # np.zeros((6, 1), dtype=float)

    def h(self, x):
        """Nonlinear measurement model yhat = h(x).

        TODO:
            Implement the GPS measurement model from Problem 3.
        """
        p_n = x.item(0)
        p_e = x.item(1)
        V = x.item(2)
        chi = x.item(3)
        c_n = x.item(4)
        c_e = x.item(5)        


        # Placeholder so the template runs before completion.
        h = np.zeros((4, 1))

        v_n = V*np.cos(chi) + c_n
        v_e = V*np.sin(chi) + c_e

        #I'm really not sure what the lower case v added to h is in the problem statement
        h[0] = p_n
        h[1] = p_e
        h[2] = np.sqrt(v_n**2+v_e**2)
        h[3] = np.atan2(v_e,v_n)
        return h # np.zeros((4, 1), dtype=float)

    def A_jacobian(self, x, u_in):
        """Jacobian A = df/dx.

        TODO:
            Implement analytic Jacobian for f.
        """
        A = np.zeros((6, 6), dtype=float)
        # df/dx, how the function change for each variable changing
        eps = .01
        
        state_dot = self.f(x, u_in )
        for i in range(len(x)):
                states_eps = x.copy()
                states_eps[i] += eps
                states_perturbed = self.f(states_eps, u_in)
                A[:,i] = ((states_perturbed-state_dot) / eps).flatten()

        return A
    
    def B_jacobian(self, x, u_in): ## IS THIS NECESSARY? IDK WHY IT WOULDN'T BE BUT IT WASN'T IN THE TEMPLATE AND THE TEST NEVER ASKS ABOUT IT
        B = np.zeros((6,2))
        eps = .01
        
        state_dot = self.f(x, u_in )
        for i in range(len(x)):
                u_in_eps = u_in.copy()
                u_in_eps[i] += eps
                states_perturbed = self.f(x, u_in_eps)
                B[:,i] = ((states_perturbed-state_dot) / eps).flatten

        return B
         

    def C_jacobian(self, x): # I ADDED U_IN, i HOPE ITS NOT A PROBLEM
        """Jacobian C = dh/dx.

        TODO:
            Implement analytic Jacobian for h.
        """
        # how the non-linear meas output changes with x, as a function of your most recent x
        C = np.zeros((4, 6))

        eps = .01
        
        sensors_orig = self.h(x)
        for i in range(len(x)):
                x_eps = x.copy()
                x_eps[i] += eps
                sensors_perturbed = self.h(x_eps)
                C[:,i] = ((sensors_perturbed-sensors_orig) / eps).flatten()

        return C

    def propagate(self, u_in):
        """EKF prediction/propagation step.

        TODO:
            Use Euler propagation for xhat and covariance prediction for P.
        """
        # TODO: Implement state and covariance propagation.
        # Placeholder return so the starter file remains runnable.
        # from Algorithm 3 chapter  pg 170 and slide 35 chptr 8
        N = 20
        Tp = self.Ts/N
        for i in range(0,N):
            xhat = self.xhat +  Tp* self.f(self.xhat,u_in)
            A = self.A_jacobian(xhat, u_in)
            # B =  np.zeros((6,2)) # TODO I'm making this 0 b/c it wasn't in the form of this file, hopefully that's not wrong # self.B_jacobian(self.xhat, u_in)
            # Qu = np.zeros((6,2)) # TODO if B is non-zero, then this is for sure the wrong Q, should be Qu see pg 165?
            Ad = np.eye(6) + A * Tp + A**2 * Tp**2
            P = Ad @ self.P @ Ad.T + Tp**2 * self.Q 
        return xhat, P

    def measurement_update(self, xhat_minus, P_minus, y):
        """EKF measurement correction step.

        TODO:
            
        """
        # TODO: Implement EKF correction equations.
        # Return posterior values once implemented.
        # from slide 35 chptr 8
        #TODO wrap chi here I think, see observer.py hw
        y[-1] = wrap_angle(y[-1])
        y_hat = self.h(xhat_minus)
        Ci = self.C_jacobian(xhat_minus)
        Si = self.R + Ci @ P_minus @ Ci.T
        Li = P_minus @ Ci.T @ np.linalg.inv(Si)
        self.xhat = self.xhat + Li @ (y-y_hat)
        self.P = (np.eye(6) - Li @ Ci) @ P_minus @ (np.eye(6) - Li @ Ci).T + Li @ self.R @ Li.T

        return self.xhat, self.P

    def update(self, inp):
        """Run one EKF step.

        Input format:
            inp = [t, u, delta, pn_gps, pe_gps, Vg_gps, chi_gps]

        TODO:
            Implement the EKF update sequence for one sample.
        """
        # TODO: Write EKF update logic here.
        # if its the first step, you'll get the intial conditions
        # propogate them forward (prediction) until you get to a measurement, then update with the measurement
        
        j = 0
        N = 20 # num of steps until a meas. update
        t = inp[0]
        u_in = np.array([inp[1],inp[2]])
        # x = np.zeros((6,1)) # we'll populate this with the new states put in, plus the c_n, c_e from last time??
        # x[0:4] = inp[3:].reshap(4,1)
        # x[4:] = self.xhat[4:]
        if t == .95:
            pass # this is just to have a debug break point
        if t % 1.0 == 0: # we're at a multiple of 20
            #first find what the propogation expects the x_hats to be, that's x_hat minus
            xhat_minus, P_minus = self.propagate(u_in)
            # then get the y from our new measured state
            y = self.h(self.xhat)
            self.xhat, self.P = self.measurement_update(xhat_minus, P_minus, y ) # TODO, idk if I need to pass in something else for xhatminus and Pminus, they should just be the last ones done before the sensor update so this should be fine...
        else:
            self.xhat, self.P =self.propagate(u_in)
            
        # keep track of the GPS meas to see if its been updated

        return self.xhat
