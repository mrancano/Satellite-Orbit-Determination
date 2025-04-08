
def plot_and_save_errors(time_array, position_deltas, velocity_deltas):
    import matplotlib.pyplot as plt

    # Position error plot
    plt.figure()
    plt.plot(time_array, position_deltas, label='Position Error (m)')
    plt.title('Position Error')
    plt.xlabel('Time (s)')
    plt.ylabel('Error (m)')
    plt.legend()
    plt.grid(True)
    plt.savefig('position_error.png')

    # Velocity error plot
    plt.figure()
    plt.plot(time_array, velocity_deltas, label='Velocity Error (m/s)')
    plt.title('Velocity Error')
    plt.xlabel('Time (s)')
    plt.ylabel('Error (m/s)')
    plt.legend()
    plt.grid(True)
    plt.savefig('velocity_error.png')

def plot_and_save_stdevs(time_array, position_magnitude_std_dev, vel_magnitude_std_dev):
    import matplotlib.pyplot as plt

    # Position error plot
    plt.figure()
    plt.plot(time_array, position_magnitude_std_dev, label='Position Standard Deviation (m)')
    plt.title('Position Standard Deviation')
    plt.xlabel('Time (s)')
    plt.ylabel('Standard Deviation (m)')
    plt.legend()
    plt.grid(True)
    plt.savefig('position_std_dev.png', dpi=300, bbox_inches='tight')

    # Velocity error plot
    plt.figure()
    plt.plot(time_array, vel_magnitude_std_dev, label='Velocity Standard Deviation (m/s)')
    plt.title('Velocity Standard Deviation')
    plt.xlabel('Time (s)')
    plt.ylabel('Standard Deviation (m/s)')
    plt.legend()
    plt.grid(True)
    plt.savefig('velocity_std_dev.png', dpi=300, bbox_inches='tight')

def plot_and_save_comparison_of_errors(time_array, pos_err, vel_err, pos_err_p, vel_err_p):
    import matplotlib.pyplot as plt

    plt.figure()
    plt.plot(time_array, pos_err, label='Position Error (Original)')
    plt.plot(time_array, pos_err_p, label='Position Error (With Noise)')
    plt.title('Position Error Comparison')
    plt.xlabel('Time (s)')
    plt.ylabel('Error (m)')
    plt.yscale('log')
    plt.legend()
    plt.grid(True)
    plt.savefig('compare_position_error.png', dpi=300, bbox_inches='tight')

    plt.figure()
    plt.plot(time_array, vel_err, label='Velocity Error (Original)')
    plt.plot(time_array, vel_err_p, label='Velocity Error (With Noise)')
    plt.title('Velocity Error Comparison')
    plt.xlabel('Time (s)')
    plt.ylabel('Error (m/s)')
    plt.legend()
    plt.grid(True)
    plt.savefig('compare_velocity_error.png', dpi=300, bbox_inches='tight')

def plot_and_save_comparison_of_stdevs(time_array, pos_std, vel_std, pos_std_p, vel_std_p):
    import matplotlib.pyplot as plt

    plt.figure()
    plt.plot(time_array, pos_std, label='Position Std Dev (Original)')
    plt.plot(time_array, pos_std_p, label='Position Std Dev (With Noise)')
    plt.title('Position Std Dev Comparison')
    plt.xlabel('Time (s)')
    plt.ylabel('Standard Deviation (m)')
    plt.legend()
    plt.grid(True)
    plt.savefig('compare_position_std_dev.png', dpi=300, bbox_inches='tight')

    plt.figure()
    plt.plot(time_array, vel_std, label='Velocity Std Dev (Original)')
    plt.plot(time_array, vel_std_p, label='Velocity Std Dev (With Noise)')
    plt.title('Velocity Std Dev Comparison')
    plt.xlabel('Time (s)')
    plt.ylabel('Standard Deviation (m/s)')
    plt.legend()
    plt.grid(True)
    plt.savefig('compare_velocity_std_dev.png', dpi=300, bbox_inches='tight')

def plot_rms_vs_position_error(time_array, rms_observation_residuals, position_errors):
    import matplotlib.pyplot as plt

    plt.figure()
    plt.plot(time_array, rms_observation_residuals, label='RMS Observation Residuals (m)')
    plt.plot(time_array, position_errors, label='Position Error (m)')
    plt.title('RMS Observation Residuals vs Position Error')
    plt.xlabel('Time (s)')
    plt.ylabel('Magnitude (m)')
    plt.legend()
    plt.grid(True)
    plt.savefig('rms_vs_position_error.png', dpi=300, bbox_inches='tight')

if __name__ == '__main__':
    import dynamics_and_state_transition as dst
    import numpy as np
    import matplotlib.pyplot as plt
    from get_data import load_data

    data = load_data()
    t, CA_range, PRN_ID, rx_gps, ry_gps, rz_gps, vx_gps, vy_gps, vz_gps, rx, ry, rz,vx,vy,vz = data

    # Create an instance of the dynamics class
    dyn = dst.dynamics()

    # Define the state vector
    x1 = np.array([rx[0],ry[0],rz[0],vx[0],vy[0],vz[0]],dtype=float)
    state_transition_matrix1 = np.eye(6) # Initial state transition matrix

    t1 = t[0]
    t2 = t[1]

    x2, state_transition_matrix2 = dyn.propagate(x1,state_transition_matrix1,t1,dyn.all_RHS,t2)

    # State Covariance Matrix
    P1 = np.zeros((6, 6))

    # Fill the diagonal elements
    np.fill_diagonal(P1, [0.002**2, 0.002**2, 0.002**2, 0.0001**2, 0.0001**2, 0.0001**2])

    # Fill the specified off-diagonal elements
    P1[0, 3] = P1[3, 0] = 0.0001*0.002*0.7
    P1[1, 4] = P1[4, 1] = 0.0001*0.002*0.7
    P1[2, 5] = P1[5, 2] = 0.0001*0.002*0.7

    #Observation Covariance Matrix
    zi = CA_range[0][CA_range[0] != 0]

    Ri = np.zeros((np.size(zi),np.size(zi)))
    np.fill_diagonal(Ri,0.003**2)

    #Process Noise Covariance Matrix
    sigma_a = 0.002 #km 
    sigma_b = 0.0002 #km/s
    Q = np.zeros((6,6))
    Q[:3,:3] = np.eye(3)*sigma_a**2
    Q[3:,3:] = np.eye(3)*sigma_b**2

    Q = np.zeros((6,6))



    #dyn.KalmanFilter_Update_Step(x1,P1,Ri,CA_range[1][CA_range[1]!= 0 ],rx_gps[1],ry_gps[1],rz_gps[1])

    estimated_states,state_covariances,observation_residuals = dyn.KalmanFilter_loop(t[0:],x1,P1,Q,CA_range[0:],rx_gps[0:],ry_gps[0:],rz_gps[0:])

    estimated_position = estimated_states[:,0:3]
    estimated_velocity = estimated_states[:,3:6]

    variances = np.diagonal(state_covariances,axis1=1,axis2=2)
    pos_variances = variances[:,0:3]
    vel_variances = variances[:,3:6]
    
    position_magnitude_std_dev = np.sqrt(np.sum(pos_variances,axis=1))*1000 #in meters
    vel_magnitude_std_dev = np.sqrt(np.sum(vel_variances,axis=1))*1000 # in meters/s

    print(f"Estimated Position at Epoch 10: {estimated_position[9]}")
    print(f"Estimated Position at Epoch 20: {estimated_position[19]}")
    print(f"Estimated Position at Epoch 30: {estimated_position[29]}")

    print(f"Estimated Velocity at Epoch 10: {estimated_velocity[9]}")
    print(f"Estimated Velocity at Epoch 20: {estimated_velocity[19]}")
    print(f"Estimated Velocity at Epoch 30: {estimated_velocity[29]}")

    precise_position = np.array([rx[0:],ry[0:],rz[0:]]).T
    precise_velocity = np.array([vx[0:],vy[0:],vz[0:]]).T

    position_deltas = np.linalg.norm(estimated_position - precise_position,axis=1)*1000 #in meters
    velocity_deltas = np.linalg.norm(estimated_velocity - precise_velocity,axis=1)*1000 # in meters/s

    plot_and_save_errors(t[:], position_deltas[:], velocity_deltas[:])
    plot_and_save_stdevs(t[:], position_magnitude_std_dev[:], vel_magnitude_std_dev[:])


    print(f"x1: {x1}")
    formatted_matrix = np.array2string(state_transition_matrix2, formatter={'float_kind':lambda x: f"{x:.10f}"})
    print(f"STM1: {formatted_matrix}")

    Q[:3,:3] = np.eye(3)*sigma_a**2
    Q[3:,3:] = np.eye(3)*sigma_b**2

    estimated_states_p,state_covariances_p,observation_residuals_p = dyn.KalmanFilter_loop(t[0:],x1,P1,Q,CA_range[0:],rx_gps[0:],ry_gps[0:],rz_gps[0:])

    estimated_position_p = estimated_states_p[:,0:3]
    estimated_velocity_p = estimated_states_p[:,3:6]

    variances_p = np.diagonal(state_covariances_p,axis1=1,axis2=2)
    pos_variances_p = variances_p[:,0:3]
    vel_variances_p = variances_p[:,3:6]
    
    position_magnitude_std_dev_p = np.sqrt(np.sum(pos_variances_p,axis=1))*1000 #in meters
    vel_magnitude_std_dev_p = np.sqrt(np.sum(vel_variances_p,axis=1))*1000 # in meters/s

    position_deltas_p = np.linalg.norm(estimated_position_p - precise_position,axis=1)*1000 #in meters
    velocity_deltas_p = np.linalg.norm(estimated_velocity_p - precise_velocity,axis=1)*1000 # in meters/s

    plot_and_save_comparison_of_errors(t[:], position_deltas[:], velocity_deltas[:], position_deltas_p[:], velocity_deltas_p[:])

    plot_and_save_comparison_of_stdevs(t[:], position_magnitude_std_dev[:], vel_magnitude_std_dev[:], position_magnitude_std_dev_p[:], vel_magnitude_std_dev_p[:])

    rms_observation_residuals_p = []
    for residual_vector in observation_residuals_p:
        valid_residuals = residual_vector[residual_vector != 0]
        if valid_residuals.size == 0:
            rms_observation_residuals_p.append(0.0)
        else:
            rms_value = np.sqrt(np.mean(valid_residuals**2))
            rms_observation_residuals_p.append(rms_value*1000) # in meters

    plot_rms_vs_position_error(t[:], rms_observation_residuals_p[:], position_deltas_p[:])