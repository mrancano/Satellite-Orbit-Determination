import os
import numpy as np

def load_data():
    # Define the base directory
    base_dir = os.path.join(os.path.dirname(__file__), 'Data_Assignment_3_with_corrections_applied')

    # Load data from text files using relative paths
    t = np.loadtxt(os.path.join(base_dir, 't.txt')) # epochs - s

    CA_range = np.loadtxt(os.path.join(base_dir, 'CA_range.txt')) # pseudorange observations from CA code - km

    PRN_ID = np.loadtxt(os.path.join(base_dir, 'PRN_ID.txt')) # PRN ID of tracked GPS satellites

    rx_gps = np.loadtxt(os.path.join(base_dir, 'rx_gps.txt')) # GPS satellite positions (transmitters) - km
    ry_gps = np.loadtxt(os.path.join(base_dir, 'ry_gps.txt'))
    rz_gps = np.loadtxt(os.path.join(base_dir, 'rz_gps.txt'))

    vx_gps = np.loadtxt(os.path.join(base_dir, 'vx_gps.txt')) # GPS satellite velocities (transmitters) - km/s
    vy_gps = np.loadtxt(os.path.join(base_dir, 'vy_gps.txt'))
    vz_gps = np.loadtxt(os.path.join(base_dir, 'vz_gps.txt'))

    rx = np.loadtxt(os.path.join(base_dir, 'rx.txt')) # precise positions (receivers) - km
    ry = np.loadtxt(os.path.join(base_dir, 'ry.txt'))
    rz = np.loadtxt(os.path.join(base_dir, 'rz.txt'))

    vx = np.loadtxt(os.path.join(base_dir, 'vx.txt')) # precise velocities (receivers) - km/s
    vy = np.loadtxt(os.path.join(base_dir, 'vy.txt'))
    vz = np.loadtxt(os.path.join(base_dir, 'vz.txt'))

    return t, CA_range, PRN_ID, rx_gps, ry_gps, rz_gps, vx_gps, vy_gps, vz_gps, rx, ry, rz, vx, vy, vz

