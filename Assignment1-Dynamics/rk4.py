import numpy as np

def RK4(y_n,f,h):
    """
    Implements the Runge-Kutta 4th-order method to solve an ODE system.
    
    Parameters:
    y_n: list
        The state vector at time t_n, [v, r].
    f: function
        The function returning derivatives [dv/dt, dr/dt].
    h: float
        The time step size.
    
    Returns:
    y_nh: list
        The updated state vector at time t_n + h.
    """
    # Compute k1
    k1 = f(y_n, 0)
    k1_v, k1_r = np.array(k1[0]), np.array(k1[1])
    
    # Compute k2
    y_k2 = [y_n[0] + 0.5 * h * k1_v, y_n[1] + 0.5 * h * k1_r]
    k2 = f(y_k2, 0)
    k2_v, k2_r = np.array(k2[0]), np.array(k2[1])
    
    # Compute k3
    y_k3 = [y_n[0] + 0.5 * h * k2_v, y_n[1] + 0.5 * h * k2_r]
    k3 = f(y_k3, 0)
    k3_v, k3_r = np.array(k3[0]), np.array(k3[1])
    
    # Compute k4
    y_k4 = [y_n[0] + h * k3_v, y_n[1] + h * k3_r]
    k4 = f(y_k4, 0)
    k4_v, k4_r = np.array(k4[0]), np.array(k4[1])
    
    # Update the state vector
    v_nh = y_n[0] + (h / 6) * (k1_v + 2 * k2_v + 2 * k3_v + k4_v)
    r_nh = y_n[1] + (h / 6) * (k1_r + 2 * k2_r + 2 * k3_r + k4_r)
    
    y_nh = [v_nh, r_nh]
    
    return y_nh