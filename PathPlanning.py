import math
from numpy.linalg import norm
from scipy.integrate import solve_ivp
import numpy as np
import matplotlib.pyplot as plt

def sv_from_coe(coe, mu):
    """
    This function computes the state vector (r, v) from the classical orbital elements (coe).
    
    Parameters:
    coe: list or numpy array
        Orbital elements [a, e, RA, incl, w, TA]
        a = semi-major axis (km)
        e = eccentricity
        RA = right ascension of the ascending node (rad)
        incl = inclination of the orbit (rad)
        w = argument of perigee (rad)
        TA = true anomaly (rad)
    
    Returns:
    r: numpy array
        Position vector in geocentric equatorial frame (km)
    v: numpy array
        Velocity vector in geocentric equatorial frame (km/s)
    """
    
    a = coe[0]
    e = coe[1]
    RA = coe[2]
    incl = coe[3]
    w = coe[4]
    TA = coe[5]
    
    # calc h
    p = a*(1-e**2)
    h = np.sqrt(mu*p)
    
    # Position and velocity vectors in the perifocal frame
    rp = (h**2 / mu) * (1 / (1 + e * np.cos(TA))) * (np.cos(TA) * np.array([1, 0, 0]) + np.sin(TA) * np.array([0, 1, 0]))
    vp = (mu / h) * (-np.sin(TA) * np.array([1, 0, 0]) + (e + np.cos(TA)) * np.array([0, 1, 0]))
    
    # Rotation matrices
    R3_W = np.array([[np.cos(RA), np.sin(RA), 0],
                     [-np.sin(RA), np.cos(RA), 0],
                     [0, 0, 1]])

    R1_i = np.array([[1, 0, 0],
                     [0, np.cos(incl), np.sin(incl)],
                     [0, -np.sin(incl), np.cos(incl)]])

    R3_w = np.array([[np.cos(w), np.sin(w), 0],
                     [-np.sin(w), np.cos(w), 0],
                     [0, 0, 1]])
    
    # Transformation matrix from perifocal to geocentric equatorial frame
    Q_pX = R3_W.T @ R1_i.T @ R3_w.T
    
    # Position and velocity in geocentric equatorial frame
    r = Q_pX @ rp
    v = Q_pX @ vp
    
    # Return row vectors
    return r, v
def cw (dr0, dv0, t, n):
    phi_rr = np.array([[4-3*np.cos(n*t), 0, 0],
                      [6*(np.sin(n*t)-n*t), 1, 0],
                      [0, 0, np.cos(n*t)]])
    phi_rv = np.array([[(1/n)*np.sin(n*t), (2/n)*(1-np.cos(n*t)), 0],
                      [(2/n)*(np.cos(n*t)-1), (1/n)*(4*np.sin(n*t)-3*n*t), 0],
                      [0, 0, (1/n)*np.sin(n*t)]])
    phi_vr = np.array([[3*n*np.sin(n*t), 0, 0],
                      [6*n*(np.cos(n*t)-1), 0, 0],
                      [0, 0, -n*np.sin(n*t)]])
    phi_vv = np.array([[np.cos(n*t), 2*np.sin(n*t), 0],
                      [-2*np.sin(n*t), 4*np.cos(n*t)-3, 0],
                      [0, 0, np.cos(n*t)]])
    dr = np.dot(phi_rr,dr0) + np.dot(phi_rv,dv0)
    dv = np.dot(phi_vr,dr0) + np.dot(phi_vv,dv0)

    return ([dr, dv])    
def cw_calc_dv0(dr0,dr1,t,n):
    phi_rr = np.array([[4-3*np.cos(n*t), 0, 0],
                       [6*(np.sin(n*t)-n*t), 1, 0],
                       [0, 0, np.cos(n*t)]])
    phi_rv = np.array([[(1/n)*np.sin(n*t), (2/n)*(1-np.cos(n*t)), 0],
                        [(2/n)*(np.cos(n*t)-1), (1/n)*(4*np.sin(n*t)-3*n*t), 0],
                        [0, 0, (1/n)*np.sin(n*t)]])
    phi_vr = np.array([[3*n*np.sin(n*t), 0, 0],
                        [6*n*(np.cos(n*t)-1), 0, 0],
                        [0, 0, -n*np.sin(n*t)]])
    phi_vv = np.array([[np.cos(n*t), 2*np.sin(n*t), 0],
                        [-2*np.sin(n*t), 4*np.cos(n*t)-3, 0],
                        [0, 0, np.cos(n*t)]])
    dv0 = np.dot(np.linalg.inv(phi_rv), dr1 - np.dot(phi_rr,dr0))
    return dv0

R_Earth = 6378
J2 = 0.00108263

# Target ICs
a = 400 + 6378 # semi major axis
I = np.deg2rad(0)
e = 0
f = np.deg2rad(0)
RAAN = np.deg2rad(0)
AOP = np.deg2rad(0)
mu = 398600 # Earth gravitational param
tau = np.sqrt(a**3 * 4 * np.pi**2 / mu) # orbital period
n = 2*np.pi / tau # mean motion
t = int(0.05*tau); # time for to dock
# t, dt = 1000, 0.01  # Total time and step size
# ts = np.arange(0, t, dt)
ts = np.linspace(0,t,t+1)
t_span = (0, t+1)

rT_ECI0, vT_ECI0 = sv_from_coe([a, e, RAAN, I, AOP, f], mu) # state vector of target sc, initially

f_x = 10*1e-3 # forces/unit mass to be applied. km/sec^2
f_y = 0  
f_z = 0  

# Chaser ICs
x0 = -1.5
y0 = 0
z0 = 0
dx0 = 0
dy0 = 0
dz0 = 0

#######################################
#######################################
####### Path Planning Function ########
#######################################
#######################################

dr0 = np.array([1.5,0,0])
dv0 = np.array([0,0,0])
dt = 0.1

NumWPs = 3
dr1 = np.array([1,0.2,0.1])
dr2 = np.array([0.3,0.1,0.05])
dr3 = np.array([0,0,0])
t01 = 500
t12 = 200
t23 = 500
drvec = np.array([dr1,dr2,dr3])
tvec = np.array([t01,t12,t23])

dv0plus = cw_calc_dv0(dr0,dr1,t01,n)
deltav01 = dv0plus - dv0
t01s = np.arange(0,t01,dt)

tmax = sum(tvec)
ts = np.arange(0,tmax,dt)
deltavs = np.zeros([NumWPs,3])
drs = np.zeros([int(tmax/dt + 1),3])
dvs = np.zeros([int(tmax/dt + 1),3])

dr0_ii = dr0
dv0minus_ii = dv0
dr_index = 0
for ii in range(0,NumWPs):
    t_ii = tvec[ii] # journey length for this waypoint
    dr1_ii = drvec[ii,:] # target waypoint
    dv0plus_ii = cw_calc_dv0(dr0_ii,dr1_ii,t_ii,n) # required velocity to reach the waypoint
    deltavs[ii] = dv0plus_ii - dv0minus_ii
    for t in np.arange(0,t_ii,dt): # this goes up to t = t_ii - dt
        dr_t, dv_t = cw(dr0_ii,dv0plus_ii,t,n)
        print(dr_t)
        drs[dr_index,:] = dr_t
        dvs[dr_index,:] = dv_t
        dr_index = dr_index + 1
    dr_t, dv_t = cw(dr0_ii,dv0plus_ii,t_ii,n)  # the state at t = t_ii, to find dv0_plus (overlap time)
    dv0minus_ii = dv_t
    dr0_ii = dr1_ii
    

#######################################
#######################################
########## Plotting ###################
#######################################
#######################################



# plot in lvlh:
fig = plt.figure(figsize=(8, 4))
ax1 = fig.add_subplot(1,2,1, projection='3d')
ax1.scatter(0, 0, 0,s=100, marker='.', c='k', label='origin')
ax1.plot(drs[:,0], drs[:,1], drs[:,2], c='g', label="from LVLH" )
for ii in range(0,NumWPs): ax1.plot(drvec[ii,0],drvec[ii,1],drvec[ii,2], 'b.')
ax1.set_xlabel('x (r-bar), m')
ax1.set_ylabel('y (v-bar), m')

ax1.set_title("lvlh frame, {} orbits".format(round(t/tau,2)))
plt.grid(True)
ax1.legend()
plt.show()