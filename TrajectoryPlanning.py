import math
from numpy.linalg import norm
from scipy.integrate import solve_ivp
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

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
def is_inside(edges, xp, yp):
    cnt = 0
    for edge in edges:
        (x1, y1), (x2, y2) = edge
        if (yp < y1) != (yp < y2) and xp < x1 + ((yp - y1) / (y2 - y1)) * (x2 - x1):
            cnt += 1
    return cnt % 2 == 1

def is_inside_2d(edges, xp, yp):
    """Checks if a 2D point (xp, yp) is inside a polygon defined by edges using ray-casting."""
    cnt = 0
    for edge in edges:
        (x1, y1), (x2, y2) = edge
        if (yp < y1) != (yp < y2) and xp < x1 + ((yp - y1) / (y2 - y1)) * (x2 - x1):
            cnt += 1
    return cnt % 2 == 1
def check_points_in_pyramid(points: np.ndarray, base, apex) -> np.ndarray:
    """
    Checks which points are inside a 3D pyramid.

    Parameters:
    - points (np.ndarray): NumPy array of shape (N, 3) with (x, y, z) points.
    - base (list): List of (x, y, z) tuples defining the base of the pyramid.
    - apex (tuple): (x, y, z) coordinates of the pyramid's apex.

    Returns:
    - NumPy array (N, 4), where the first three columns are (x, y, z),
      and the fourth column is 1 (inside) or 0 (outside).
    """
    if points.shape[1] != 3:
        raise ValueError("Input points array must have shape (N, 3)")

    # Define 2D projections
    xy_triangle = [(base[0][0], base[0][1]), (base[1][0], base[1][1]), (apex[0], apex[1])]
    zx_triangle = [(base[1][0], base[1][2]), (base[2][0], base[2][2]), (apex[0], apex[2])]

    # Convert to edge lists for 2D checking
    xy_edges = list(zip(xy_triangle, xy_triangle[1:] + [xy_triangle[0]]))
    zx_edges = list(zip(zx_triangle, zx_triangle[1:] + [zx_triangle[0]]))

    # Initialize results array with an extra column for inside/outside check
    results = np.zeros((points.shape[0], 4))
    results[:, :3] = points  # Copy x, y, z values

    # Check each point
    for i, (x, y, z) in enumerate(points):
        inside_xy = is_inside_2d(xy_edges, x, y)
        inside_zx = is_inside_2d(zx_edges, x, z)
        results[i, 3] = int(inside_xy and inside_zx)  # Store 1 (inside) or 0 (outside)

    return results
def plot_pyramid_with_points(base, apex, results,ax):
    """
    Plots a 3D pyramid and its 2D projections (XY and ZX) with the given points.
    
    Parameters:
    - base: List of (x, y, z) tuples defining the base.
    - apex: (x, y, z) coordinates of the pyramid's apex.
    - results: NumPy array containing (x, y, z, inside) points.
    """
    # Define pyramid faces
    faces = [
        [base[0], base[1], apex],  # Front face
        [base[1], base[2], apex],  # Right face
        [base[2], base[3], apex],  # Back face
        [base[3], base[0], apex],  # Left face
        base  # Base
    ]

    # Extract 2D projections
    xy_triangle = [(base[0][0], base[0][1]), (base[1][0], base[1][1]), (apex[0], apex[1])]
    zx_triangle = [(base[1][0], base[1][2]), (base[2][0], base[2][2]), (apex[0], apex[2])]
    
    # ---- 3D Plot ----
    ax.add_collection3d(Poly3DCollection(faces, alpha=0.3, edgecolor='k',label="Approach Corridor"))

    # Plot points
    for x, y, z, inside in results:
        if inside == False:
            ax.scatter(x, y, z, '.', c='r', s=5, label = "Outside Approach Corridor")

    fig = plt.figure(figsize=(8, 4))
    # ---- XY Projection ----
    ax2 = fig.add_subplot(122)
    xy_xs, xy_ys = zip(*xy_triangle)
    ax2.fill(xy_xs, xy_ys, "b", alpha=0.5)

    for x, y, _, inside in results:
        ax2.scatter(x, y, color='g' if inside else 'r', s=5)

    ax2.set_xlabel("X")
    ax2.set_ylabel("Y")
    ax2.set_title("XY Projection")
    ax2.set_aspect("equal")

    # ---- ZX Projection ----
    ax3 = fig.add_subplot(222)
    zx_xs, zx_zs = zip(*zx_triangle)
    ax3.fill(zx_xs, zx_zs, "r", alpha=0.5)

    for x, _, z, inside in results:
        ax3.scatter(x, z, color='g' if inside else 'r', s=5)

    ax3.set_xlabel("X")
    ax3.set_ylabel("Z")
    ax3.set_title("ZX Projection")
    ax3.set_aspect("equal")


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
t = int(0.005*tau); # time for to dock
# t, dt = 1000, 0.01  # Total time and step size
# ts = np.arange(0, t, dt)
ts = np.linspace(0,t,t+1)
t_span = (0, t+1)

rT_ECI0, vT_ECI0 = sv_from_coe([a, e, RAAN, I, AOP, f], mu) # state vector of target sc, initially
f_x = 10*1e-3 # forces/unit mass to be applied. km/sec^2
f_y = 0  
f_z = 0  

# Chaser ICs
x0 = -1.49
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

dr0 = np.array([1.49,0,0])
dv0 = np.array([0,0,0])
dt = 10

NumWPs = 5
dr1 = np.array([1,0,0])
dr2 = np.array([0.8,-0.07,-0.06])
dr3 = np.array([0.3,0.02,0.01])
dr4 = np.array([0.1,0,0])
dr5 = np.array([0.04,0.4,0.4])
t01 = 300
t12 = 200
t23 = 400
t34 = 100
t45 = 50

drvec = np.array([dr1,dr2,dr3,dr4,dr5])
tvec = np.array([t01,t12,t23,t34,t45])

def PlanTrajectory(NumWPs, drvec, tvec, dr0, dv0, dt):
    # this creates a trajectory (position, drs, and time, ts) that follows a set of waypoints (drvec)
    # each waypoint, drvec(n), takes time t_n-1_n to complete
    # a container for the position (drs), velocities (dvs), const interval time (ts) and total deltaV is retuned
    
    tmax = sum(tvec)
    ts = np.arange(0,tmax,dt)
    deltavs = np.zeros([NumWPs,3])
    Traj = np.zeros([int(tmax/dt + 1),7])
    dvs = np.zeros([int(tmax/dt + 1),3])

    dr0_ii = dr0
    dv0minus_ii = dv0
    dr_index = 0
    t_runningTotal = 0
    for ii in range(0,NumWPs):
        t_ii = tvec[ii] # journey length for this waypoint
        dr1_ii = drvec[ii,:] # target waypoint
        dv0plus_ii = cw_calc_dv0(dr0_ii,dr1_ii,t_ii,n) # required velocity to reach the waypoint
        deltavs[ii] = dv0plus_ii - dv0minus_ii
        if ii == NumWPs-1: # last run
            t_ii = t_ii + 1 # to include the very final index
        for t in np.arange(0,t_ii,dt): # this goes up to t = t_ii - dt
            dr_t, dv_t = cw(dr0_ii,dv0plus_ii,t,n)
            Traj[dr_index,0:3] = dr_t
            Traj[dr_index,3:6] = dv_t
            Traj[dr_index,6] = t_runningTotal
            dr_index = dr_index + 1
            t_runningTotal = t_runningTotal + dt
        dr_t, dv_t = cw(dr0_ii,dv0plus_ii,t_ii,n)  # the state at t = t_ii, to find dv0_plus (overlap time)
        dv0minus_ii = dv_t
        dr0_ii = dr1_ii

    return Traj, deltavs

Traj, deltavs = PlanTrajectory(NumWPs, drvec, tvec, dr0, dv0, dt)


# Define Pyramid geometry
base = [(1.5, 0.2, 0.2), (1.5, -0.2, 0.2), (1.5, -0.2, -0.2), (1.5, 0.2, -0.2)]
apex = (0,0,0)

# Find points outside pyramid
results = check_points_in_pyramid(Traj[:,0:3], base, apex)

##########################################
##########################################
############# Plotting ###################
##########################################
##########################################

# plot in lvlh:
fig = plt.figure(figsize=(8, 4))
ax1 = fig.add_subplot(111, projection='3d')
ax1.scatter(0, 0, 0,s=100, marker='.', c='k', label='origin')
ax1.plot(Traj[:,0], Traj[:,1], Traj[:,2], c='g', label="from LVLH" )
ax1.plot(drvec[:,0],drvec[:,1],drvec[:,2], 'bo', label="Waypoint")
ax1.set_xlabel('x (r-bar), m')
ax1.set_ylabel('y (v-bar), m')
ax1.set_zlabel('z (h-bar), m')

ax1.set_title("lvlh frame, {} orbits".format(round(t/tau,2)))
plt.grid(True)
ax1.legend()
plot_pyramid_with_points(base, apex, results,ax1)
plt.show()