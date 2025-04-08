import numpy as np
from Assignment_2_get_data import load_data


def measured_pseudorange_observation_equation(parameter_vec,x_transmiters,y_transmitters,z_transmitters,transmitter_clock_correction):
    #parameter_vec [rx,ry,rz,receiver_clock_correction]
    #transmitter_array = rows x,y,z of transmitter positions columns are different transmitters
    #transmitter_clock_correction = 1 row columns are different transmitters

    c = 299792458/1000  # speed of light in km/s

    r_transmitter_array = np.array([x_transmiters,y_transmitters,z_transmitters]) #3 rows 12 columns

    r_receiver_array = np.tile(parameter_vec[0:3], (r_transmitter_array.shape[1], 1)).T  # rows x,y,z of receiver position columns are repeated receiver positions
    
    receiver_clock_correction = parameter_vec[-1]

    measured_pseudoranges = np.linalg.norm(r_receiver_array-r_transmitter_array,axis=0)+c*(receiver_clock_correction)-c*(transmitter_clock_correction)


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

    return H

if __name__ == "__main__":
    data = load_data()
    t, CA_range, PRN_ID, clk_gps, rx_gps, ry_gps, rz_gps, vx_gps, vy_gps, vz_gps, rx, ry, rz = data



    x_0 = np.array([rx[0],ry[0],rz[0],-0.00707162]) #what should the receiver clock offset be at x_0?

    y_0 = measured_pseudorange_observation_equation(x_0,rx_gps[0],ry_gps[0],rz_gps[0],clk_gps[0])
    


    #print(y_0-CA_range[0])


    H = H_measured_pseudorange(x_0,rx_gps[0],ry_gps[0],rz_gps[0],clk_gps[0])

    y = y_0 + H @ (x_0-np.array([rx[0],ry[0],rz[0],0]))

    print(f"x_0: {x_0}")
    print(f"y_0: {y_0}")
    print(f"H: {H}")

    #print(f"y:{}")


    #H_approx =(measured_pseudorange_observation_equation(x_0+np.array([0,0.0001,0,0]),rx_gps[0],ry_gps[0],rz_gps[0],clk_gps[0])-y_0)/0.0001
    #print(H[:,1]-H_approx)


    