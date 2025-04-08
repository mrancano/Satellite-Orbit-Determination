import numpy as np
from Assignment_2_get_data import load_data
import matplotlib.pyplot as plt


def measured_pseudorange_observation_equation(parameter_vec,x_transmiters,y_transmitters,z_transmitters,transmitter_clock_correction):
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

def H_measured_pseudorange(parameter_vec,x_transmiters,y_transmitters,z_transmitters,transmitter_clock_correction):
    c = 299792458/1000 # speed of light in km/s

    r_transmitter_array = np.array([x_transmiters,y_transmitters,z_transmitters]) #3 rows 12 columns

    r_receiver_array = np.tile(parameter_vec[0:3], (r_transmitter_array.shape[1], 1)).T  # rows x,y,z of receiver position columns are repeated receiver positions
    
    receiver_clock_correction = parameter_vec[-1]

    position_component_differences = (r_receiver_array-r_transmitter_array).T #sample row: x_r-x_t,y_r-y_t,z_r-z_t



    position_vector_differences = np.tile(np.linalg.norm(r_receiver_array-r_transmitter_array,axis=0),(3,1)).T


    H = np.zeros((12,4))

    H[:,0:3] = position_component_differences/position_vector_differences
    H[:,3] = c

    valid_indices = x_transmiters != 0
    H = H[valid_indices]

    return H

def iterative_non_linear_least_squares(parameter_vec,observation_vec,x_gps,y_gps,z_gps,clk_gps,W_yy):
    #parameter vect is x_0 (initial guess)
    #x_gps,y_gps,z_gps,clk_gps are the gps positions and time corrections at the epoch t
    #W_yy is the inverse of the covariance matrix of the pseudorange measurements
    #observation_vec is the pseudorange measurements at the epoch t

    tolerance = 1e-6
    max_iterations = 100
    iteration = 0

    while iteration < max_iterations:
        fx_0 = measured_pseudorange_observation_equation(parameter_vec, x_gps, y_gps, z_gps, clk_gps)
 
        dy = observation_vec - fx_0

        H = H_measured_pseudorange(parameter_vec, x_gps, y_gps, z_gps, clk_gps)

        inverse_matrix = np.linalg.inv(H.T @ W_yy @ H)

        dx = inverse_matrix @ H.T @ W_yy @ dy

        x_hat = parameter_vec + dx

        if np.linalg.norm(dx) < tolerance:
            break

        parameter_vec = x_hat
        iteration += 1
    #print(iteration)

    covariance_matrix = pseudo_range_var*inverse_matrix #variance of the observations

    return x_hat,dy,covariance_matrix

def estimate_all_epochs_uncorrected(CA_range, clk_gps, rx_gps, ry_gps, rz_gps):

    x_at_all_epochs_uncorrected = np.ones((np.shape(CA_range)[0],4))*100
    x_at_all_epochs_uncorrected[:,3] = 0
    dy_at_all_epochs_uncorrected = np.zeros((np.shape(CA_range)[0],np.shape(CA_range)[1]))
    covariance_matrices_uncorrected = np.zeros((np.shape(CA_range)[0],4,4))

    
    

    for i in range(np.shape(CA_range)[0]):
        x_0 = x_at_all_epochs_uncorrected[i]
        y = CA_range[i][CA_range[i] != 0]
        P_yy = np.ones((np.size(y),np.size(y)))*1.8
        np.fill_diagonal(P_yy,pseudo_range_var)
        W_yy = pseudo_range_var*np.linalg.inv(P_yy)

        x_at_all_epochs_uncorrected[i],dy_at_all_epochs_uncorrected[i][CA_range[i] != 0], covariance_matrices_uncorrected[i] = iterative_non_linear_least_squares(x_0,y,rx_gps[i],ry_gps[i],rz_gps[i],clk_gps[i],W_yy)
        #x_at_all_epochs_corrected[i][0:3] = x_at_all_epochs_uncorrected[i][0:3]-np.array([vx[i],vy[i],vz[i]])*clk[i]

   

    return x_at_all_epochs_uncorrected,dy_at_all_epochs_uncorrected

def estimate_all_epochs_corrected(CA_range, clk_gps, rx_gps, ry_gps, rz_gps,vx_gps,vy_gps,vz_gps):
    #rx_gps,ry_gps,rz_gps are the gps positions at the epoch t
    #we will correct them for light time effect to to get them at the epoch t-tao
    c = 299792458/1000  # speed of light in km/s
    w = 7.292115e-5 # Earth's rotation rate in rad/s


    x_at_all_epochs_corrected = np.ones((np.shape(CA_range)[0],4))*100
    x_at_all_epochs_corrected[:,3] = 0
    dy_at_all_epochs_corrected = np.zeros((np.shape(CA_range)[0],np.shape(CA_range)[1]))
    covariance_matrices_corrected = np.zeros((np.shape(CA_range)[0],4,4))
    
    for i in range(np.shape(CA_range)[0]):
        x_0 = x_at_all_epochs_corrected[i]
        y = CA_range[i][CA_range[i] != 0]
        P_yy = np.ones((np.size(y),np.size(y)))*0.2*0.003*0.003
        np.fill_diagonal(P_yy,pseudo_range_var)
        W_yy = pseudo_range_var*np.linalg.inv(P_yy)


        #light time effect correction
        tao = CA_range[i]/c  
        
        rx = rx_gps[i] - vx_gps[i]*tao
        ry = ry_gps[i] - vy_gps[i]*tao
        rz = rz_gps[i] - vz_gps[i]*tao

        vx = vx_gps[i]
        vy = vy_gps[i]
        vz = vz_gps[i]

        rotation_matrices = np.array([
            [
                [np.cos(t * w), np.sin(t * w), 0],
                [-np.sin(t * w), np.cos(t * w), 0],
                [0, 0, 1]
            ] for t in tao
        ])

        rx, ry, rz = np.einsum("ijk,ik->ij", rotation_matrices, np.array([rx, ry, rz]).T).T

    



        #relativistic effect correction
        delta_t_rel_array = -2/c**2*np.matmul(np.array([rx,ry,rz]).T,np.array([vx_gps[i],vy_gps[i],vz_gps[i]]))

        delta_t_rel = np.diag(delta_t_rel_array)

        clk_gps[i] = clk_gps[i]+delta_t_rel

        

        x_at_all_epochs_corrected[i],dy_at_all_epochs_corrected[i][CA_range[i] != 0], covariance_matrices_corrected[i] = iterative_non_linear_least_squares(x_0,y,rx,ry,rz,clk_gps[i],W_yy)
        #x_at_all_epochs_corrected[i][0:3] = x_at_all_epochs_uncorrected[i][0:3]-np.array([vx[i],vy[i],vz[i]])*clk[i]




    

    return x_at_all_epochs_corrected,dy_at_all_epochs_corrected, covariance_matrices_corrected

def plot_position_differences(estimated_positions, precise_positions, time,j=0,PDOP = False,PDOP_list = None):
    differences = estimated_positions - precise_positions
    norms = np.linalg.norm(differences, axis=1)

    plt.figure(figsize=(10, 6))
    plt.plot(time, norms*1000,label="Position Difference")
    if PDOP:
        plt.plot(time,np.array(PDOP_list)*1000,label=r"PDOP for $\sigma$ = 3m")
        plt.legend()
    plt.title("Difference between Estimated Positions and Precise Orbit", fontsize=14)
    plt.xlabel("Time (s)", fontsize=12)
    plt.ylabel("Position Difference (m)", fontsize=12)
    plt.grid(True)
    plt.savefig(f"estimated_vs_precise_orbit{j}.png", dpi=300)
    plt.close()

def plot_receiver_clock_offset(time, receiver_clock_offset):
    plt.figure(figsize=(10, 6))
    plt.plot(time, receiver_clock_offset)
    plt.title("Receiver Clock Offset", fontsize=14)
    plt.xlabel("Time (s)", fontsize=12)
    plt.ylabel("Clock Offset (s)", fontsize=12)
    plt.grid(True)
    plt.savefig("receiver_clock_offset.png", dpi=300)
    plt.close()


def plot_residuals(dy_uncorrected,dy_corrected,time):
    dy_uncorrected_norm = np.linalg.norm(dy_uncorrected,axis=1)*1000
    dy_corrected_norm = np.linalg.norm(dy_corrected,axis=1)*1000

    plt.figure(figsize=(10, 6))
    plt.plot(time, dy_uncorrected_norm, label="Uncorrected")
    plt.plot(time, dy_corrected_norm, label="Corrected")
    plt.title("Residuals", fontsize=14)
    plt.xlabel("Time (s)", fontsize=12)
    plt.ylabel("Magnitude of Residuals Vector (m)", fontsize=12)
    plt.legend()
    plt.grid(True)
    plt.savefig("residuals.png", dpi=300)
    plt.close()



if __name__ == "__main__":
    data = load_data()
    t, CA_range, PRN_ID, clk_gps, rx_gps, ry_gps, rz_gps, vx_gps, vy_gps, vz_gps, rx, ry, rz,vx,vy,vz = data

    #clk_receiver = np.ones((np.size(t),))*0.0056

    pseudo_range_var = 0.003**2

    x_0 = np.array([rx[0],ry[0],rz[0],-0.00707162]) #what should the receiver clock offset be at x_0?
   
    y = CA_range[1][CA_range[1] != 0]

    P_yy = np.ones((np.size(y),np.size(y)))*0.2*0.003*0.003
    np.fill_diagonal(P_yy,pseudo_range_var)
    W_yy = np.linalg.inv(P_yy)

    x_hat = iterative_non_linear_least_squares(x_0,y,rx_gps[1],ry_gps[1],rz_gps[1],clk_gps[1],W_yy)


    x_hat_vec,dy_vec_uncorrected = estimate_all_epochs_uncorrected(CA_range, clk_gps, rx_gps, ry_gps, rz_gps)

    x_hat_vec_corrected ,dy_vec_corrected, covariance_matrices_corrected = estimate_all_epochs_corrected(CA_range, clk_gps, rx_gps, ry_gps, rz_gps,vx_gps,vy_gps,vz_gps)


    clk_receiver_estimated_vec = x_hat_vec[:,3]
    clk_receiver_estimated_vec_corrected = x_hat_vec_corrected[:,3]

    precise_positions = np.array([rx, ry, rz]).T 
    precise_velocities = np.array([vx, vy, vz]).T

    adjusted_precise_positions = precise_positions - precise_velocities * clk_receiver_estimated_vec[:, np.newaxis]


    sum_residuals = np.sum(np.linalg.norm(x_hat_vec[:,:3] - adjusted_precise_positions, axis=1))
    sum_residuals_corrected = np.sum(np.linalg.norm(x_hat_vec_corrected[:,:3] - adjusted_precise_positions, axis=1))
    mean_residuals_corrected = sum_residuals_corrected / len(t)

    print(sum_residuals)
    print(sum_residuals_corrected)
    PDOP_list = []
    for i in covariance_matrices_corrected:
        PDOP_list.append(np.sqrt(i[0][0]+i[1][1]+i[2][2]))

    
   

    plot_position_differences(x_hat_vec[:, 0:3],adjusted_precise_positions, t+clk_receiver_estimated_vec)
    plot_position_differences(x_hat_vec_corrected[:, 0:3],adjusted_precise_positions, t+clk_receiver_estimated_vec_corrected,j=1,PDOP=True,PDOP_list=PDOP_list)

    plot_receiver_clock_offset(t, clk_receiver_estimated_vec_corrected)

    plot_residuals(dy_vec_uncorrected,dy_vec_corrected,t)

    print(np.linalg.norm(precise_positions[0])-6371)
    
    # print(t[0]-t[0]+clk_receiver_estimated_vec_corrected[0])
    # print(x_hat_vec_corrected[0][:3])
    # print(t[1]-t[0]+clk_receiver_estimated_vec_corrected[1])
    # print(x_hat_vec_corrected[1][:3])
    # print(t[2]-t[0]+clk_receiver_estimated_vec_corrected[2])
    # print(x_hat_vec_corrected[2][:3])
    # print(t[3]-t[0]+clk_receiver_estimated_vec_corrected[3])
    # print(x_hat_vec_corrected[3][:3])


    print(mean_residuals_corrected*1000)
    