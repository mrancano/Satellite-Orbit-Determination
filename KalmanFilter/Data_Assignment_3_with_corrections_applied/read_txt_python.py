import numpy as np

t = np.loadtxt('t.txt') # epochs - s

CA_range = np.loadtxt('CA_range.txt') # pseudorange observations from CA code - km

PRN_ID = np.loadtxt('PRN_ID.txt') # PRN ID of tracked GPS satellites

rx_gps = np.loadtxt('rx_gps.txt') # GPS satellite positions (transmitters) - km
ry_gps = np.loadtxt('ry_gps.txt')
rz_gps = np.loadtxt('rz_gps.txt')

vx_gps = np.loadtxt('vx_gps.txt') # GPS satellite velocities (transmitters) - km/s
vy_gps = np.loadtxt('vy_gps.txt')
vz_gps = np.loadtxt('vz_gps.txt')

rx = np.loadtxt('rx.txt') # precise positions (receivers) - km
ry = np.loadtxt('ry.txt')
rz = np.loadtxt('rz.txt')

vx = np.loadtxt('vx.txt') # precise velocities (receivers) - km/s
vy = np.loadtxt('vy.txt')
vz = np.loadtxt('vz.txt')

