import numpy as np
import scipy as sp


class dynamics:
    def __init__(self,we = 7.292115e-05,GM = 3.986004415e5):
        self.we = we
        self.GM = GM
        #CONSTANTS (ASSUMED CONSTANT)
        c = 299792.458 # speed of light, km/s
        GM = 3.986004415e5 # Earth’s gravitational constant, km3/s2
        R = 6378.13660 # Earth reference radius, km
        C20 = -4.841692151273e-04 # gravity field coefficient, dimensionless
        we = 7.292115e-05 # Earth rotation rate, rad/s
        CD = 2.6 # drag coefficient, dimensionless
        rho = 1e-2 # atmospheric density, kg/km3
        A = 1e-6 # cross-section area, km2
        m = 500 # satellite mass, kg

    def measured_pseudorange_observation_equation(self,parameter_vec,x_transmiters,y_transmitters,z_transmitters,transmitter_clock_correction):
        #parameter_vec [rx,ry,rz,receiver_clock_correction]
        #transmitter_array = rows x,y,z of transmitter positions columns are different transmitters
        #transmitter_clock_correction = 1 row columns are different transmitters

        c = 299792458/1000  # speed of light in km/s

        r_transmitter_array = np.array([x_transmiters,y_transmitters,z_transmitters]) #3 rows 12 columns

        r_receiver_array = np.tile(parameter_vec[0:3], (r_transmitter_array.shape[1], 1)).T  # rows x,y,z of receiver position columns are repeated receiver positions
        
        receiver_clock_correction = parameter_vec[-1]

        measured_pseudoranges = np.linalg.norm(r_receiver_array-r_transmitter_array,axis=0)+c*(receiver_clock_correction)-c*(transmitter_clock_correction)

        valid_indices = x_transmiters != 0
        measured_pseudoranges = measured_pseudoranges[valid_indices]

        return measured_pseudoranges

    def H_measured_pseudorange(self,parameter_vec,x_transmiters,y_transmitters,z_transmitters):
        c = 299792458/1000 # speed of light in km/s

        r_transmitter_array = np.array([x_transmiters,y_transmitters,z_transmitters]) #3 rows 12 columns

        r_receiver_array = np.tile(parameter_vec[0:3], (r_transmitter_array.shape[1], 1)).T  # rows x,y,z of receiver position columns are repeated receiver positions
        
        receiver_clock_correction = parameter_vec[-1]

        position_component_differences = (r_receiver_array-r_transmitter_array).T #sample row: x_r-x_t,y_r-y_t,z_r-z_t



        position_vector_differences = np.tile(np.linalg.norm(r_receiver_array-r_transmitter_array,axis=0),(3,1)).T

        H = np.zeros((len(x_transmiters),len(parameter_vec))) #10 by 4 in this case

        H[:,0:3] = position_component_differences/position_vector_differences
        H[:,3] = c

        valid_indices = x_transmiters != 0
        H = H[valid_indices]

        return H

    def f(self,t,x,p = None):
        # x: state vector (x,y,z,vx,vy,vz)
        # p: parameter vector
        # t: time
        # return: xdot
        we = self.we # Earth rotation rate, rad/s
        GM = self.GM # Earth’s gravitational constant, km3/s2

        r_vec = x[:3]
        v_vec = x[3:]

        #Earths gravity
        r_hat = r_vec/np.linalg.norm(r_vec)
        a_rot = -GM/(np.dot(r_vec,r_vec))*r_hat 

        #Coreolis and Centrifugal
        we_vec = np.array([0,0,we])
        Omega = np.zeros((3,3))
        Omega[0,1] = we
        Omega[1,0] = -we

        a_cor = 2*np.matmul(Omega,v_vec)
        a_centrifugal = -np.matmul(Omega,np.matmul(Omega,r_vec))
        a_centrifugal = -np.matmul(Omega,Omega).dot(r_vec)

        fx = np.zeros_like(x)
        fx[:3] = v_vec
        fx[3:] = a_rot + a_cor + a_centrifugal

        return fx

    def dfdx_matrix(self,t,x,p = None):
        # x: state vector (x,y,z,vx,vy,vz)
        # p: parameter vector
        # t: time
        # return: dfdx 6x6 matrix
        GM = self.GM # Earth’s gravitational constant, km3/s2
        we = self.we # Earth rotation rate, rad/s
        Omega = np.zeros((3,3))
        Omega[0,1] = we
        Omega[1,0] = -we

        r_vec = x[:3]
        r = np.linalg.norm(r_vec)

        dfdx = np.zeros((6,6))

        dfdx[:3,:3] = np.zeros((3,3))
        dfdx[:3,3:] = np.eye(3)

        dfdx[3:,:3] = GM*(3/r**5*np.outer(r_vec,r_vec) - 1/r**3*np.eye(3)) - np.matmul(Omega,Omega)
        dfdx[3:,3:] = 2*Omega

        return dfdx 

    def state_transition_matrix_ode_RHS(self,t,x,state_transition_matrix,p = None):
        # x: state vector (x,y,z,vx,vy,vz)
        # p: parameter vector
        # t: time
        # return: dfdx dot state_transition_matrix

        dfdx = self.dfdx_matrix(t,x,p)

        return np.matmul(dfdx,state_transition_matrix)

    def all_RHS(self,t,g, p = None):
        # g: concatenated state vector and state transition matrix
        # p: parameter vector
        # t: time
        # return: RHS of state vector ODE and state transition matrix ODE evaluated at x,t

        x = g[:6]
        state_transition_matrix = np.reshape(g[6:],(6,6))

        state_RHS = self.f(t,x,p)
        state_transition_matrix_RHS = np.reshape(self.state_transition_matrix_ode_RHS(t,x,state_transition_matrix,p),(36,))

        all_RHS = np.concatenate((state_RHS,state_transition_matrix_RHS),axis = 0)
        
        return all_RHS

    def propagate(self,x0,state_transition_matrix0,t0,f,t1,p = None):
        state_transition_matrix0 = np.reshape(state_transition_matrix0,(36,))
        g0 = np.concatenate((x0,state_transition_matrix0),axis = 0)

        integrator = sp.integrate.ode(self.all_RHS)
        integrator.set_integrator("vode", method="bdf")
        integrator.set_initial_value(g0,t0)
        if p is not None:
            integrator.set_f_params(p)
        g1 = integrator.integrate(t1)

        x1 = g1[:6]
        state_transition_matrix1 = np.reshape(g1[6:],(6,6))

        return x1,state_transition_matrix1
    
    def KalmanFilter_Update_Step(self,x0,P0,R0,z0,x_transmiters,y_transmitters,z_transmitters):
        # x0: initial state vector
        # P0: initial state covariance matrix
        # R0: Initial Observation Covariance Matrix
        # z0: observation vector
        # return: x1, P1

        parameter_vec = np.array([x0[0],x0[1],x0[2],0])

        H0_pos = self.H_measured_pseudorange(parameter_vec,x_transmiters,y_transmitters,z_transmitters)[:,:3] #only position columns (excluding clock correction because it is already taken into account)

        H0 = np.zeros((np.shape(H0_pos)[0],np.shape(H0_pos)[1]+3))
        H0[:,:3] = H0_pos


        hx0 = self.measured_pseudorange_observation_equation(parameter_vec,x_transmiters,y_transmitters,z_transmitters,0)

        dz0 = z0 - hx0

        right_matrix = np.linalg.inv(np.matmul(H0,np.matmul(P0,H0.T))+R0)

        K0 = np.matmul(P0,np.matmul(H0.T,right_matrix))

        x_hat0 = x0 + np.matmul(K0,dz0)

        #print(np.matmul(K0,dz0))

        P_hat0 = (np.eye(np.shape(P0)[0])-np.matmul(K0,H0)).dot(P0)

        e0 = dz0 - np.matmul(H0,np.matmul(K0,dz0))


        #print(f"x_hat0: {x_hat0}")
        #print(f"P_hat0: {P_hat0}")

        return x_hat0,P_hat0,e0

    def KalmanFilter_Propagate_Step(self,x0,P0,t0,t1,Q = np.zeros((6,6))):

        phi00 = np.eye(6)

        x1,phi10 = self.propagate(x0,phi00,t0,self.all_RHS,t1)

        P1 = np.matmul(phi10,np.matmul(P0,phi10.T))+Q


        return x1,P1
    


    def KalmanFilter_loop(self,epochs,x0,P0,Q,CA_range, rx_gps, ry_gps, rz_gps):
        # x0: initial state vector
        # P0: initial state covariance matrix
        # CA_range: CA range matrix
        # rx_gps: x gps position array
        # ry_gps: y gps position array
        # rz_gps: z gps position array

        k=0

        z0 = CA_range[k][CA_range[k] != 0]

        R0 = np.zeros((np.size(z0),np.size(z0)))
        np.fill_diagonal(R0,0.003**2)

        estimated_states = np.zeros((len(epochs),6))
        state_covariances = np.zeros((len(epochs),6,6))
        observation_residuals = np.zeros((len(epochs),np.size(CA_range[k])))

        x_hat,P_hat,e_k  = self.KalmanFilter_Update_Step(x0,P0,R0,CA_range[k][CA_range[k]!= 0],rx_gps[k],ry_gps[k],rz_gps[k])

        estimated_states[k] = x_hat
        state_covariances[k] = P_hat
        observation_residuals[k][:np.size(z0)] = e_k


        while True:


            x_kplus1 , P_kplus1 = self.KalmanFilter_Propagate_Step(x_hat,P_hat,epochs[k],epochs[k+1],Q) 

            k += 1

            zk = CA_range[k][CA_range[k] != 0]

            Rk = np.zeros((np.size(zk),np.size(zk)))
            np.fill_diagonal(Rk,0.003**2)   

            x_hat,P_hat,e_k  = self.KalmanFilter_Update_Step(x_kplus1,P_kplus1,Rk,CA_range[k][CA_range[k]!= 0],rx_gps[k],ry_gps[k],rz_gps[k])
        
            estimated_states[k] = x_hat
            state_covariances[k] = P_hat
            observation_residuals[k][:np.size(zk)] = e_k

            if k == len(epochs)-1:
                break

        return estimated_states,state_covariances,observation_residuals

