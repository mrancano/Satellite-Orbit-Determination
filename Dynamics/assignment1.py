import numpy as np
import scipy as sp
import matplotlib.pyplot as plt


t = np.loadtxt('data/t.txt') # epochs - s

r = np.loadtxt('data/r.txt') # precise positions - km

v = np.loadtxt('data/v.txt') # precise velocities - km/s

def f(t,y,mode): #RHS of ODE
    #y is [vx,vy,vz,rx,ry,rz]

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

    #FUNCTION

    # Extract velocity and position
    # r is the position vector at time t
    # v is the velocity vector at time 
    v = y[:3]  # First three elements are velocity
    r = y[3:]  # Last three elements are position


    #Earths gravity
    r_hat = r/np.linalg.norm(r)
    a_rot = -GM/(np.dot(r,r))*r_hat 

    #Coreolis and Centrifugal
    we_vec = np.array([0,0,we])
    Omega = np.zeros((3,3))
    Omega[0,1] = we
    Omega[1,0] = -we


    a_cor = 2*np.matmul(Omega,v)
    #print(np.linalg.norm(a_cor))
    a_centrifugal = -np.matmul(Omega,np.matmul(Omega,r))
    #print(np.linalg.norm(a_centrifugal))

    #Earth flattening
    a_flattening = -GM/R**3*C20*np.sqrt(45)*(R/np.linalg.norm(r))**5*(r*(2.5*(r[-1]/np.linalg.norm(r))**2-0.5)-[0,0,r[-1]])
    #print(np.linalg.norm(a_flattening))
    #Atmospheric Drag

    a_drag = -0.5*A/m*CD*rho*np.linalg.norm(v)*v

    #print(np.linalg.norm(a_drag))
    #Included Accelerations
    if mode == 1:
        dvdt = a_rot
    elif mode == 2:
        dvdt = a_cor+a_centrifugal+a_rot
    elif mode == 3:
        dvdt = a_cor+a_centrifugal+a_rot + a_flattening
    elif mode == 4:
        dvdt = a_cor+a_centrifugal+a_rot + a_flattening + a_drag

    drdt = v

    return np.hstack((dvdt, drdt))


def propagate(y0,t0,f,t1,mode):
    integrator = sp.integrate.ode(f)
    integrator.set_initial_value(y0,t0)
    integrator.set_f_params(mode)
    y1 = integrator.integrate(t1)

    return y1



t0 = t[0]
y0 = np.hstack((v[0], r[0]))  # Flatten into a single array

t1 = t[1]
v1_true = v[1]
r1_true = r[1]
y1_pred = propagate(y0,t0,f,t1,1)
v1_pred = y1_pred[:3]
r1_pred = y1_pred[3:]

#print(y1_pred)



modes = [1,2,3,4]
positions_pred = np.zeros((len(modes),len(r),3))
print(positions_pred)
velocities_pred = np.zeros((len(modes),len(v),3))
for a in modes:
    for i in range(len(v)):
        y_pred = propagate(y0,t0,f,t[i],a)
        positions_pred[a-1,i] = y_pred[3:]
        velocities_pred[a-1,i] = y_pred[:3]

# Calculate errors
position_errors = np.linalg.norm(positions_pred - r, axis=2)  # Shape: (4, len(t))
velocity_errors = np.linalg.norm(velocities_pred - v, axis=2)  # Shape: (4, len(t))


# Mode descriptions
mode_labels = [
    "No perturbations, no frame rotation",
    "No perturbations, frame rotation",
    "Flattening + frame rotation",
    "Flattening + frame rotation + drag"
]

# Plot position errors
plt.figure(figsize=(10, 6))
for i, mode in enumerate(mode_labels):
    plt.plot(t[1:], position_errors[i][1:], label=mode)
plt.title("Position Errors Over Time")
plt.xlabel("Time (s)")
plt.ylabel("Position Error (km)")
plt.yscale('log')
plt.legend()
plt.grid(True)
#plt.show()

# Plot velocity errors
plt.figure(figsize=(10, 6))
for i, mode in enumerate(mode_labels):
    plt.plot(t[1:], velocity_errors[i][1:], label=mode)
plt.title("Velocity Errors Over Time")
plt.xlabel("Time (s)")
plt.ylabel("Velocity Error (km/s)")
plt.yscale('log')
plt.legend()
plt.grid(True)
#plt.show()

final_position_errors = position_errors[:, -1]


