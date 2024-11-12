from scipy.integrate import solve_ivp
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


#### --------------------------------------------------------------------------------------------------------------------------------------------------------------
# Define Functions
#### --------------------------------------------------------------------------------------------------------------------------------------------------------------
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
def draw_pyramid(w, Lx):


    # Define vertices of the pyramid
    apex = np.array([0, 0, 0])  # The apex of the pyramid at the origin
    
    # Vertices for the base (centered along the positive x-axis with length Lx)
    base_vertices = np.array([
        [Lx, -w / 2, -w / 2],  # Bottom-left
        [Lx, w / 2, -w / 2],   # Bottom-right
        [Lx, w / 2, w / 2],    # Top-right
        [Lx, -w / 2, w / 2]    # Top-left
    ])

    # Create the faces of the pyramid using the vertices
    faces = [
        [apex, base_vertices[0], base_vertices[1]],  # Side 1
        [apex, base_vertices[1], base_vertices[2]],  # Side 2
        [apex, base_vertices[2], base_vertices[3]],  # Side 3
        [apex, base_vertices[3], base_vertices[0]],  # Side 4
        [base_vertices[0], base_vertices[1], base_vertices[2], base_vertices[3]]  # Base
    ]

    # Plot the pyramid
    poly3d = Poly3DCollection(faces, facecolors='blue', linewidths=0.1, edgecolors='k', alpha=0.2)
    ax.add_collection3d(poly3d)
def plotTraj(xCWs, yCWs, zCWs, Name, colour, axi, Marker=False):
    axi.plot(xCWs, yCWs, zCWs, c=colour, label=Name)
    if Marker == True:
        axi.scatter(xCWs[-1], yCWs[-1], zCWs[-1], s=100, marker='*', c=colour, label=Name+"_END")
        axi.scatter(xCWs[0], yCWs[0], zCWs[0], s=100, marker='.', c=colour, label=Name+"_START")

    return np.void
def Hill_eqns(t, u):
    # Unpack the variables
    x, y, z, dx, dy, dz = u
    print(t)
    # Define the equations based on the image
    ddx = 2 * n * dy + 3 * n**2 * x + f_x(t)
    ddy = -2 * n * dx + f_y(t)
    ddz = -n**2 * z + f_z(t)
    
    return [dx, dy, dz, ddx, ddy, ddz] # Define the system of differential equations
def TwoBP(t, u):#, mu):
    # unpack variables
    x, y, z, dx, dy, dz = u
    r = np.array([x, y, z])
    
    # define eqns using 2BP
    ddx = (-mu/(np.linalg.norm(r)**3)) * x
    ddy = (-mu/(np.linalg.norm(r)**3)) * y
    ddz = (-mu/(np.linalg.norm(r)**3)) * z
    
    return [dx, dy, dz, ddx, ddy, ddz]
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
def LVLH2ECI(r_LVLH, v_LVLH, r_ECI_T, v_ECI_T):
    ihat = r_ECI_T/np.linalg.norm(r_ECI_T)
    h_ECI_T = np.cross(r_ECI_T, v_ECI_T)
    khat = h_ECI_T/np.linalg.norm(h_ECI_T)
    jhat = np.cross(khat, ihat)
    
    T_LVLH2ECI = np.array([ihat.T, jhat.T, khat.T]).T

    r_ECI_C = np.dot(T_LVLH2ECI, r_LVLH)
    v_ECI_C = np.dot(T_LVLH2ECI, v_LVLH)
    
    return r_ECI_C, v_ECI_C
def animate_3d_trajectories(data, framerate=30, ANIMATE=True):
    """
    Animate 3D trajectories from raw array data.
    
    Parameters:
    - data: A dictionary containing the following keys:
        - 'LVLH': (x, y, z) for the LVLH frame
        - 'ECI': ECI data
    - framerate: The desired frame rate for the animation.
    """
    if ANIMATE==True:
        # Unpack the data
        x1, y1, z1 = data['LVLH']
        x2, y2, z2, x3, y3, z3 = data['ECI']
        
        n = 200 # this shortens each array, by evenly sampling every n entry
        x1 = x1[::n]
        y1 = y1[::n]
        z1 = z1[::n]
        
        x2 = x2[::n]
        y2 = y2[::n]
        z2 = z2[::n]
         
        x3 = x3[::n]
        y3 = y3[::n]
        z3 = z3[::n]

        num_points = len(x1)  # Assuming all trajectories have the same length

        # Set up the figure and subplots
        fig = plt.figure(figsize=(12, 6))

        # Create 3D subplot for the LVLH frame  
        ax1 = fig.add_subplot(121, projection='3d')
        ax1.set_title('LVLH Frame')
        ax1.set_xlabel('X, m')
        ax1.set_ylabel('Y, m')
        ax1.set_zlabel('Z, m')

        # Create 3D subplot for the ECI Frame
        ax2 = fig.add_subplot(122, projection='3d')
        ax2.set_title('ECI Frame')
        ax2.set_xlabel('X, km')
        ax2.set_ylabel('Y, km')
        ax2.set_zlabel('Z, km')

        # Set fixed limits for the axes
        ax1.set_xlim([-np.max(abs(np.array([x1,y1,z1]))), np.max(abs(np.array([x1,y1,z1])))])
        ax1.set_ylim([-np.max(abs(np.array([x1,y1,z1]))), np.max(abs(np.array([x1,y1,z1])))])
        ax1.set_zlim([-np.max(abs(np.array([x1,y1,z1]))), np.max(abs(np.array([x1,y1,z1])))])
        
        ax2.set_xlim([-np.max(abs(np.array([x2,y2,z2]))), -np.max(abs(np.array([x2,y2,z2])))])
        ax2.set_ylim([-np.max(abs(np.array([x2,y2,z2]))), -np.max(abs(np.array([x2,y2,z2])))])
        ax2.set_zlim([-np.max(abs(np.array([x2,y2,z2]))), -np.max(abs(np.array([x2,y2,z2])))]) 

        # Initialization function
        def init():
            ax1.cla()  # Clear ax1
            ax1.set_title('LVLH Frame')
            ax1.set_xlabel('X, m')
            ax1.set_ylabel('Y, m')
            ax1.set_zlabel('Z, m')
        
            ax2.cla()  # Clear ax2
            ax2.set_title('ECI Frame')
            ax2.set_xlabel('X, km')
            ax2.set_ylabel('Y, km')
            ax2.set_zlabel('Z, km')
        
            # Reapply limits
            ax1.set_xlim([-np.max(abs(np.array([x1,y1,z1]))), np.max(abs(np.array([x1,y1,z1])))])
            ax1.set_ylim([-np.max(abs(np.array([x1,y1,z1]))), np.max(abs(np.array([x1,y1,z1])))])
            ax1.set_zlim([-np.max(abs(np.array([x1,y1,z1]))), np.max(abs(np.array([x1,y1,z1])))])
            ax2.set_xlim([-np.max(abs(np.array([x2,y2,z2]))), np.max(abs(np.array([x2,y2,z2])))])
            ax2.set_ylim([-np.max(abs(np.array([x2,y2,z2]))), np.max(abs(np.array([x2,y2,z2])))])
            ax2.set_zlim([-np.max(abs(np.array([x2,y2,z2]))), np.max(abs(np.array([x2,y2,z2])))]) 
            return []

        # Update function
        def update(frame):
            ax1.cla()  # Clear previous frame for ax1
            ax1.set_title('LVLH Frame')
            ax1.set_xlabel('X, m')
            ax1.set_ylabel('Y, m')
            ax1.set_zlabel('Z, m')
        
            # Set fixed limits
            ax1.set_xlim([-np.max(abs(np.array([x1,y1,z1]))), np.max(abs(np.array([x1,y1,z1])))])
            ax1.set_ylim([-np.max(abs(np.array([x1,y1,z1]))), np.max(abs(np.array([x1,y1,z1])))])
            ax1.set_zlim([-np.max(abs(np.array([x1,y1,z1]))), np.max(abs(np.array([x1,y1,z1])))])
        
            # Update the LVLH Frame 
            ax1.plot(x1[:frame], y1[:frame], z1[:frame], color='blue', lw=1, label='Chaser')
            if frame > 0:
                ax1.scatter(x1[frame-1], y1[frame-1], z1[frame-1], color='blue', marker='*', s=100)
            ax1.scatter(0, 0, 0, s=100, marker='.', c='k', label="Origin")


            ax2.cla()  # Clear previous frame for ax2
            ax2.set_title('ECI Frame')
            ax2.set_xlabel('X, km')
            ax2.set_ylabel('Y, km')
            ax2.set_zlabel('Z, km')
            # Set fixed limits
            ax2.set_xlim([-np.max(abs(np.array([x2,y2,z2]))), np.max(abs(np.array([x2,y2,z2])))])
            ax2.set_ylim([-np.max(abs(np.array([x2,y2,z2]))), np.max(abs(np.array([x2,y2,z2])))])
            ax2.set_zlim([-np.max(abs(np.array([x2,y2,z2]))), np.max(abs(np.array([x2,y2,z2])))])     
            # Update the ECI Frame 
            ax2.plot(x2[:frame], y2[:frame], z2[:frame], color='blue', lw=1, label='Chaser')
                # Add a star marker at the current point
            if frame > 0:
                ax2.scatter(x2[frame-1], y2[frame-1], z2[frame-1], color='blue', marker='*', s=100)
                
            ax2.plot(x3[:frame], y3[:frame], z3[:frame], color='red', lw=1, label='Target')
            if frame > 0:
                ax2.scatter(x3[frame-1], y3[frame-1], z3[frame-1], color='red', marker='*', s=100)
            ax2.scatter(0, 0, 0, s=100, marker='.', c='k', label="Origin")
            
            # Add legend for subplot 2
            ax2.legend()
        
            return []

        # Create the animation
        anim = FuncAnimation(fig, update, frames=num_points, init_func=init, interval=1000, blit=True)
        # Display the animation
        plt.show()

    return np.void

#### --------------------------------------------------------------------------------------------------------------------------------------------------------------
# Define scenario
#### --------------------------------------------------------------------------------------------------------------------------------------------------------------
a = 400 + 6378 # semi major axis
I = np.deg2rad(0)
e = 0
f = np.deg2rad(0)
RAAN = np.deg2rad(0)
AOP = np.deg2rad(0)
mu = 398600 # Earth gravitational param
tau = np.sqrt(a**3 * 4 * np.pi**2 / mu) # orbital period
n = 2*np.pi / tau # mean motion
t = int(2*tau); # time for to dock
ts = np.linspace(0,t,t+1)
t_span = (0, t+1)

rT, vT = sv_from_coe([a, e, RAAN, I, AOP, f], mu) # state vector of target sc, initially

f_x = lambda t: 0  # Define external force as a function of time
f_y = lambda t: 0  # Define external force as a function of time
f_z = lambda t: 0  # Define external force as a function of time

# Example: Relative motion on orbit with different altitude
# x0 = 100
# y0 = 0
# z0 = 0
# dx0 = 0
# dy0 = -1.5*n*x0
# dz0 = 0

#  Release at a distance from the station CoM in the z-direction
# x0 = -10
# y0 = 0
# z0 = 0
# dx0 = 0
# dy0 = 0
# dz0 = 0

#∆V in an orbital direction
# x0 = 0
# y0 = 0
# z0 = 0
# dx0 = 0
# dy0 = -0.01
# dz0 = 0

# Straight line V-bar approach with constant velocity
x0 = 0
y0 = 0
z0 = 0
dx0 = 0
dy0 = 0.01
dz0 = 0
f_x = lambda t: -2*n*dy0  # Define external force as a function of time

# Straight line R-bar approach with constant velocity
# x0 = 20
# y0 = 0
# z0 = 0
# dx0 = 0.01
# dy0 = 0
# dz0 = 0
# f_y = lambda t: 2*n*dx0  # Define external force as a function of time
# f_x = lambda t: -3*(n**2)*(dx0*t + x0)

ICs_LVLH_C = [x0, y0, z0, dx0, dy0, dz0]  # [x0, y0, z0, dx0, dy0, dz0], initial conditions
ICs_ECI_T = [rT[0], rT[1], rT[2], vT[0], vT[1], vT[2]]

#### --------------------------------------------------------------------------------------------------------------------------------------------------------------
# Solve for trajectories
#### --------------------------------------------------------------------------------------------------------------------------------------------------------------
sol_LVLH = solve_ivp(Hill_eqns, t_span, ICs_LVLH_C, t_eval=ts, method='RK45',rtol=1e-10) # Solve the system of differential equations
sol_ECI_T = solve_ivp(TwoBP, t_span, ICs_ECI_T, t_eval=ts, method='RK45',rtol=1e-10) # Solve the system of differential equations

r_LVLH_C = np.array([sol_LVLH.y[0], sol_LVLH.y[1], sol_LVLH.y[2]])
v_LVLH_C = np.array([sol_LVLH.y[3], sol_LVLH.y[4], sol_LVLH.y[5]])
r_ECI_T = np.array([sol_ECI_T.y[0], sol_ECI_T.y[1], sol_ECI_T.y[2]])
v_ECI_T = np.array([sol_ECI_T.y[3], sol_ECI_T.y[4], sol_ECI_T.y[5]])


r_ECI_C = np.zeros(r_LVLH_C.shape)
v_ECI_C = np.zeros(r_LVLH_C.shape)
ii = 0
for t in ts:
    r_ECI_Cii, v_ECI_Cii = LVLH2ECI(r_LVLH_C[:,ii], v_LVLH_C[:,ii], r_ECI_T[:,ii], v_ECI_T[:,ii])
    r_ECI_C[:,ii] = (r_ECI_Cii/1000 + r_ECI_T[:,ii]) # all in km now
    v_ECI_C[:,ii] = (v_ECI_Cii/1000 + v_ECI_T[:,ii])
    ii = ii + 1


print(r_ECI_C)
#### --------------------------------------------------------------------------------------------------------------------------------------------------------------
# Plotting
# #### --------------------------------------------------------------------------------------------------------------------------------------------------------------
# Plot in LVLH:
fig = plt.figure(figsize=(8, 4))
ax1 = fig.add_subplot(1,2,1)
ax1.scatter(0, 0, s=100, marker='.', c='k', label='Origin')
ax1.plot(sol_LVLH.y[0], sol_LVLH.y[1], c='b')

ax1.set_xlabel('X (R-bar), m')
ax1.set_ylabel('Y (V-bar), m')

# ax1 = fig.add_subplot(1,2,1, projection='3d')
# ax1.scatter(0, 0, 0, s=100, marker='.', c='k', label="Origin")

# plotTraj(sol_LVLH.y[0], sol_LVLH.y[1], sol_LVLH.y[2], "LVLH", 'b', ax1)

# ax1.set_xlabel('X, m')
# ax1.set_ylabel('Y, m')
# ax1.set_zlabel('Z, m')
# ax1.tick_params(axis='x', labelsize=10)
# ax1.tick_params(axis='y', labelsize=10)
# ax1.tick_params(axis='z', labelsize=10)

# ax1.set_xlim([-np.max(abs(r_LVLH_C[0,:])), np.max(abs(r_LVLH_C[0,:]))])
# ax1.set_ylim([-np.max(abs(r_LVLH_C)), np.max(abs(r_LVLH_C))])
# ax1.set_zlim([-np.max(abs(r_LVLH_C)), np.max(abs(r_LVLH_C))])
ax1.set_title("LVLH Frame, {} orbits".format(round(t/tau,2)))
plt.grid(True)
ax1.legend()


# #### --------------------------------------------------------------------------------------------------------------------------------------------------------------
# # Plot in ECI:
ax2 = fig.add_subplot(1,2,2, projection='3d')
plotTraj(sol_ECI_T.y[0], sol_ECI_T.y[1], sol_ECI_T.y[2], "Target", 'r', ax2, Marker=True)
plotTraj(r_ECI_C[0], r_ECI_C[1], r_ECI_C[2], "Chaser", 'g', ax2, Marker=True)
ax2.scatter(0, 0, 0, s=100, marker='.', c='k', label="Origin")

ax2.set_xlabel('X, km')
ax2.set_ylabel('Y, km')
ax2.set_zlabel('Z, km')
ax2.tick_params(axis='x', labelsize=10)
ax2.tick_params(axis='y', labelsize=10)
ax2.tick_params(axis='z', labelsize=10)

ax2.set_title("ECI, {} orbits".format(round(t/tau,2)))
ax2.legend()
plt.show()



#### --------------------------------------------------------------------------------------------------------------------------------------------------------------
# Animate both plots
data = {
    'LVLH': (sol_LVLH.y[0], sol_LVLH.y[1], sol_LVLH.y[2]),
    'ECI': (r_ECI_C[0], r_ECI_C[1], r_ECI_C[2], sol_ECI_T.y[0], sol_ECI_T.y[1], sol_ECI_T.y[2])
}
animate_3d_trajectories(data, framerate=200, ANIMATE=False)
