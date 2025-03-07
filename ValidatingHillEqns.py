import math
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
    r_C_LVLHii = np.array([x,y,z])
    v_C_LVLHii = np.array([dx,dy,dz])
    
    # Define the equations based on the image
    ddx = 2 * n * dy + 3 * n**2 * x + f_x(t)
    ddy = -2 * n * dx + f_y(t)
    ddz = -n**2 * z + f_z(t)
    a = np.array([ddx, ddy, ddz])
    
    # if J2_BOOL == True:
    #     print(t)

    #     r_T_ECIii = np.array([rvt_T_ECI[0,np.where(rvt_T_ECI[6,:] == t)[0]][0], rvt_T_ECI[1,np.where(rvt_T_ECI[6,:] == t)[0]][0], rvt_T_ECI[2,np.where(rvt_T_ECI[6,:] == t)[0]][0]]) 
    #     v_T_ECIii = np.array([rvt_T_ECI[3,np.where(rvt_T_ECI[6,:] == t)[0]][0], rvt_T_ECI[4,np.where(rvt_T_ECI[6,:] == t)[0]][0], rvt_T_ECI[5,np.where(rvt_T_ECI[6,:] == t)[0]][0]])
    #     r_C_ECIii, v_C_ECIii = LVLH2ECI(r_T_ECIii, v_T_ECIii, r_C_LVLHii, v_C_LVLHii)

    #     rmag = np.linalg.norm(r_C_ECIii)
    #     z2 = z*z
    #     tx = (x/rmag) * (5 * (z2/rmag**2) - 1)
    #     ty = (y/rmag) * (5 * (z2/rmag**2) - 1)
    #     tz = (z/rmag) * (5 * (z2/rmag**2) - 3)
        
    #     a_j2 = (1.5*J2*mu*(R_Earth**2)/(rmag**4)) * np.array([tx,ty,tz]) # orbital mechanics for engineering students, 4th edition. this was verified against example 10.2 in this textbook
       
    #     w = np.array([1,1,n])
        
    #     a_j2_LVLH = a_j2 - (np.cross(w, np.cross(w, r_C_LVLHii)) + 2*np.cross(w, v_C_LVLHii))
        
    #     a = a + a_j2_LVLH
    #     print(a.shape)
    return [dx, dy, dz, a[0], a[1], a[2]] # Define the system of differential equations
def Orbital_DiffyQ(t, u):#, mu):
    # unpack variables
    x, y, z, dx, dy, dz = u
    r = np.array([x, y, z])
    
    # 2 body acceleration
    ddx = (-mu/(np.linalg.norm(r)**3)) * x
    ddy = (-mu/(np.linalg.norm(r)**3)) * y
    ddz = (-mu/(np.linalg.norm(r)**3)) * z
    a = np.array([ddx, ddy, ddz])
    
    if J2_BOOL == True:
        rmag = np.linalg.norm(r)
        z2 = z*z
        tx = (x/rmag) * (5 * (z2/rmag**2) - 1)
        ty = (y/rmag) * (5 * (z2/rmag**2) - 1)
        tz = (z/rmag) * (5 * (z2/rmag**2) - 3)
        
        a_j2 = (1.5*J2*mu*(R_Earth**2)/(rmag**4)) * np.array([tx,ty,tz]) # orbital mechanics for engineering students, 4th edition. this was verified against example 10.2 in this textbook
        a = a + a_j2
            
    return [dx, dy, dz, a[0], a[1], a[2]]

def Combined_DiffyQ(t, u):
    # unpack variables
    xT_ECI, yT_ECI, zT_ECI, dxT_ECI, dyT_ECI, dzT_ECI, x_ECI, y_ECI, z_ECI, dx_ECI, dy_ECI, dz_ECI, x_LVLH, y_LVLH, z_LVLH, dx_LVLH, dy_LVLH, dz_LVLH = u

    # Target 2 body accel
    rT_ECI = np.array([xT_ECI, yT_ECI, zT_ECI])
    ddxT_ECI = (-mu/(np.linalg.norm(rT_ECI)**3)) * xT_ECI
    ddyT_ECI = (-mu/(np.linalg.norm(rT_ECI)**3)) * yT_ECI
    ddzT_ECI = (-mu/(np.linalg.norm(rT_ECI)**3)) * zT_ECI
    aT_ECI = np.array([ddxT_ECI, ddyT_ECI, ddzT_ECI])

    # 2 body acceleration
    r_ECI = np.array([x_ECI, y_ECI, z_ECI])
    ddx_ECI = (-mu/(np.linalg.norm(r_ECI)**3)) * x_ECI 
    ddy_ECI = (-mu/(np.linalg.norm(r_ECI)**3)) * y_ECI
    ddz_ECI = (-mu/(np.linalg.norm(r_ECI)**3)) * z_ECI
    a_ECI = np.array([ddx_ECI, ddy_ECI, ddz_ECI])
   
    # rmag_ECI = np.linalg.norm(r_ECI)
    # z2_ECI = z_ECI*z_ECI
    # tx = (x_ECI/rmag_ECI) * (5 * (z2_ECI/rmag_ECI**2) - 1)
    # ty = (y_ECI/rmag_ECI) * (5 * (z2_ECI/rmag_ECI**2) - 1)
    # tz = (z_ECI/rmag_ECI) * (5 * (z2_ECI/rmag_ECI**2) - 3)
    # a_j2_ECI = (1.5*J2*mu*(R_Earth**2)/(rmag_ECI**4)) * np.array([tx,ty,tz]) # orbital mechanics for engineering students, 4th edition. this was verified against example 10.2 in this textbook
    # a_ECI = a_ECI + a_j2_ECI
    # f_ECI = a_j2_ECI*1000 # km --> m !!

    
    # Hill Eqns
    r_LVLH = np.array([x_LVLH,y_LVLH,z_LVLH])
    v_LVLH = np.array([dx_LVLH,dy_LVLH,dz_LVLH])
    ddx_LVLH = 2 * n * dy_LVLH + 3 * n**2 * x_LVLH + f_x(t)
    ddy_LVLH = -2 * n * dx_LVLH + f_y(t)
    ddz_LVLH = -n**2 * z_LVLH + f_z(t)
    a_LVLH = np.array([ddx_LVLH, ddy_LVLH, ddz_LVLH])
    # f_LVLH, ignore = ECI2LVLH(rT_ECI, np.array([dxT_ECI, dyT_ECI, dzT_ECI]), f_ECI, f_ECI )
    # a_LVLH = a_LVLH + f_LVLH
    # print(a_LVLH)
    # # J2
    # if J2_BOOL == True:
    #     # Target ECI
    #     # rmagT_ECI = np.linalg.norm(rT_ECI)
    #     # z2T_ECI = zT_ECI*zT_ECI
    #     # tx = (xT_ECI/rmagT_ECI) * (5 * (z2T_ECI/rmagT_ECI**2) - 1)
    #     # ty = (yT_ECI/rmagT_ECI) * (5 * (z2T_ECI/rmagT_ECI**2) - 1)
    #     # tz = (zT_ECI/rmagT_ECI) * (5 * (z2T_ECI/rmagT_ECI**2) - 3)
    #     # aT_j2_ECI = (1.5*J2*mu*(R_Earth**2)/(rmagT_ECI**4)) * np.array([tx,ty,tz]) # orbital mechanics for engineering students, 4th edition. this was verified against example 10.2 in this textbook
    #     # aT_ECI = aT_ECI + aT_j2_ECI        

    #     # ECI
    #     rmag_ECI = np.linalg.norm(r_ECI)
    #     z2_ECI = z_ECI*z_ECI
    #     tx = (x_ECI/rmag_ECI) * (5 * (z2_ECI/rmag_ECI**2) - 1)
    #     ty = (y_ECI/rmag_ECI) * (5 * (z2_ECI/rmag_ECI**2) - 1)
    #     tz = (z_ECI/rmag_ECI) * (5 * (z2_ECI/rmag_ECI**2) - 3)
    #     a_j2_ECI = (1.5*J2*mu*(R_Earth**2)/(rmag_ECI**4)) * np.array([tx,ty,tz]) # orbital mechanics for engineering students, 4th edition. this was verified against example 10.2 in this textbook
    #     a_ECI = a_ECI + a_j2_ECI
        
    #     # LVLH
    #     # w = np.array([0,0,n])
    #     # a_j2_LVLH = a_j2_ECI - ( aT_ECI + np.cross(w, np.cross(w, r_LVLH)) + 2*np.cross(w, v_LVLH) )
    #     # a_LVLH = a_LVLH + a_j2_LVLH

    #     # trying just a simple rotation matrix ?
    #     a_j2_LVLH, ignore = ECI2LVLH(rT_ECI, np.array([dxT_ECI,dyT_ECI,dzT_ECI]), a_j2_ECI, a_j2_ECI ) # i THINK this is correct. this matches up with the Cornell video. you can convert external forces by dotting it along the respective unit vectors of the LVLH frame        
    #     a_LVLH = a_LVLH + a_j2_LVLH
        
    return [ dxT_ECI,dyT_ECI,dzT_ECI,aT_ECI[0],aT_ECI[1],aT_ECI[2], dx_ECI,dy_ECI,dz_ECI,a_ECI[0],a_ECI[1],a_ECI[2],dx_LVLH,dy_LVLH,dz_LVLH,a_LVLH[0],a_LVLH[1],a_LVLH[2] ]

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
def coe_from_sv(R, V, mu):
    """
    Computes the classical orbital elements (coe) from the state vector (R, V).
    
    Parameters:
    R (numpy array): Position vector in the geocentric equatorial frame (km)
    V (numpy array): Velocity vector in the geocentric equatorial frame (km/s)
    mu (float): Gravitational parameter (km^3/s^2)
    
    Returns:
    coe (list): Vector of orbital elements [h, e, RA, incl, w, TA, a]
    """
    eps = 1.e-10
    
    r = np.linalg.norm(R)
    v = np.linalg.norm(V)
    vr = np.dot(R, V) / r
    H = np.cross(R, V)
    h = np.linalg.norm(H)
    
    # Inclination (rad)
    incl = np.arccos(H[2] / h)
    
    # Node line vector
    N = np.cross([0, 0, 1], H)
    n = np.linalg.norm(N)
    
    # Right ascension of the ascending node (RA) (rad)
    if n != 0:
        RA = np.arccos(N[0] / n)
        if N[1] < 0:
            RA = 2 * np.pi - RA
    else:
        RA = 0
    
    # Eccentricity vector
    E = (1 / mu) * ((v**2 - mu / r) * R - r * vr * V)
    e = np.linalg.norm(E)
    
    # Argument of perigee (w) (rad)
    if n != 0:
        if e > eps:
            w = np.arccos(np.dot(N, E) / (n * e))
            if E[2] < 0:
                w = 2 * np.pi - w
        else:
            w = 0
    else:
        w = 0
    
    # True anomaly (TA) (rad)
    if e > eps:
        TA = np.arccos(np.dot(E, R) / (e * r))
        if vr < 0:
            TA = 2 * np.pi - TA
    else:
        cp = np.cross(N, R)
        if cp[2] >= 0:
            TA = np.arccos(np.dot(N, R) / (n * r))
        else:
            TA = 2 * np.pi - np.arccos(np.dot(N, R) / (n * r))
    
    # Semi-major axis (a) (km)
    a = h**2 / mu / (1 - e**2)
    
    # Orbital elements
    #coe = [h, e, RA, incl, w, TA, a] # dont care about h
    coe = [e, RA, incl, w, TA, a]
    
    return coe # verified using check case in orbital mechanics for engineering students
def cw_docking_v0(r0, t, n):
    # this calculates a v0 so that the trajectory docks (reaches r = [0,0,0] at time t)
    phi_rr = np.array([[4-3*np.cos(n*t), 0, 0],
                      [6*(np.sin(n*t)-n*t), 1, 0],
                      [0, 0, np.cos(n*t)]])
    phi_rv = np.array([[(1/n)*np.sin(n*t), (2/n)*(1-np.cos(n*t)), 0],
                      [(2/n)*(np.cos(n*t)-1), (1/n)*(4*np.sin(n*t)-3*n*t), 0],
                      [0, 0, (1/n)*np.sin(n*t)]])
    v0 = -np.dot(np.linalg.inv(phi_rv), np.dot(phi_rr,r0))
    
    return v0

def ECI2LVLH(r1_I, v1_I, r12_I, v12_I):
    # 1: target satellite
    # 2: chaser satellite
    # 12: chaser relative to target satellite
    # I: inertial frame (ECI)
    # R: rotating frame (LVLH)
    # this converts a postion and velocity vector from eci to lvlh frame

    h1_I = np.cross(r1_I,v1_I)
    h1_Ihat = h1_I/np.linalg.norm(h1_I)
    ex_hat = r1_I / np.linalg.norm(r1_I)
    ez_hat = h1_Ihat
    ey_hat = np.cross(ez_hat, ex_hat)
    ex_hat_dot = (1/np.linalg.norm(r1_I)) * (v1_I - np.dot(r1_I/np.linalg.norm(r1_I), v1_I)*r1_I/np.linalg.norm(r1_I))
    ez_hat_dot = np.array([0,0,0])
    ey_hat_dot = np.cross(ez_hat,ex_hat_dot) + np.cross(ez_hat_dot,ex_hat)
    C = np.array([ex_hat,ey_hat,ez_hat])
    C_dot = np.array([ex_hat_dot, ey_hat_dot, ez_hat_dot])

    r12_R = np.dot(C, r12_I)
    v12_R = np.dot(C_dot, r12_I) + np.dot(C,v12_I)
    
    return r12_R, v12_R # verified, using test case in "degenerate conic page"
def LVLH2ECI(r1_I, v1_I, r12_R, v12_R):
    # 1: target satellite
    # 2: chaser satellite
    # 12: chaser relative to target satellite
    # I: inertial frame (ECI)
    # R: rotating frame (LVLH)
    # this converts a position and velocity vector from the lvlh to the eci frame

    h1_I = np.cross(r1_I,v1_I)
    h1_Ihat = h1_I/np.linalg.norm(h1_I)
    ex_hat = r1_I / np.linalg.norm(r1_I)
    ez_hat = h1_Ihat
    ey_hat = np.cross(ez_hat, ex_hat)
    ex_hat_dot = (1/np.linalg.norm(r1_I)) * (v1_I - np.dot(r1_I/np.linalg.norm(r1_I), v1_I)*r1_I/np.linalg.norm(r1_I))
    ez_hat_dot = np.array([0,0,0])
    ey_hat_dot = np.cross(ez_hat,ex_hat_dot) + np.cross(ez_hat_dot,ex_hat)
    C = np.array([ex_hat,ey_hat,ez_hat])
    C_dot = np.array([ex_hat_dot, ey_hat_dot, ez_hat_dot])

    r12_I = np.dot(C.T, r12_R)
    v12_I = np.dot(C_dot.T, r12_R) + np.dot(C.T,v12_R)
    
    return r12_I, v12_I # verified, using test case in "degenerate conic page"

# def accelECI2LVLH(t, r_LVLH, v_LVLH):
#     rii_ECI = 


#     w = np.array([[1,0,0],
#                   [0,1,0],
#                   [0,0,n]])
#     a_LVLH = a_ECI - (np.cross(w, np.cross(w, r_LVLH)) + 2*np.cross(w, v_LVLH))
#     return a_LVLH

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
def round_to_sig_figs(number, sig_figs):
    if number == 0:
        return 0  # Special case for zero
    else:
        # Calculate the order of magnitude of the number
        magnitude = math.floor(math.log10(abs(number)))
        # Scale the number to the required significant figures
        factor = 10 ** (sig_figs - 1 - magnitude)
        return round(number * factor) / factor
    
#### --------------------------------------------------------------------------------------------------------------------------------------------------------------
# Define scenario
#### --------------------------------------------------------------------------------------------------------------------------------------------------------------
J2_BOOL = False
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
t = int(10*60); # time for to dock
ts = np.linspace(0,t,t+1)
t_span = (0, t+1)

rT_ECI0, vT_ECI0 = sv_from_coe([a, e, RAAN, I, AOP, f], mu) # state vector of target sc, initially

f_x = lambda t: 0  # Define external force as a function of time
f_y = lambda t: 0  # Define external force as a function of time
f_z = lambda t: 0  # Define external force as a function of time


# Chaser ICs
x0 = 1.5
y0 = 0.4
z0 = 0.4
dx0 = 0
dy0 = 0
dz0 = 0


v0 = cw_docking_v0(np.array([x0,y0,z0]), t, n)
dx0 = v0[0]
dy0 = v0[1]
dz0 = v0[2]

rC_LVLH0 = np.array([x0, y0, z0])
vC_LVLH0 = np.array([dx0, dy0, dz0])

rCrel_ECI0, vCrel_ECI0 = LVLH2ECI(rT_ECI0, vT_ECI0, rC_LVLH0, vC_LVLH0)
rC_ECI0 = rCrel_ECI0/1000 + rT_ECI0 # chaser position in ECI, in km
vC_ECI0 = vCrel_ECI0/1000 + vT_ECI0 # in km/sec

# ICs for solve_ivp
ICs_LVLH_C = [rC_LVLH0[0], rC_LVLH0[1], rC_LVLH0[2], vC_LVLH0[0], vC_LVLH0[1], vC_LVLH0[2]]
ICs_ECI_T = [rT_ECI0[0], rT_ECI0[1], rT_ECI0[2], vT_ECI0[0], vT_ECI0[1], vT_ECI0[2]]
ICs_ECI_C = [rC_ECI0[0], rC_ECI0[1], rC_ECI0[2], vC_ECI0[0], vC_ECI0[1], vC_ECI0[2]]

ICs_COMB_C = [rT_ECI0[0], rT_ECI0[1], rT_ECI0[2], vT_ECI0[0], vT_ECI0[1], vT_ECI0[2], rC_ECI0[0], rC_ECI0[1], rC_ECI0[2], vC_ECI0[0], vC_ECI0[1], vC_ECI0[2], rC_LVLH0[0], rC_LVLH0[1], rC_LVLH0[2], vC_LVLH0[0], vC_LVLH0[1], vC_LVLH0[2]]

print("Chaser in LVLH: {} m, {} m/sec\nChaser in ECI: {} km, {} km/sec\nTarget in ECI: {} km, {} km/sec".format(rC_LVLH0, vC_LVLH0, rC_ECI0, vC_ECI0, rT_ECI0, vT_ECI0))


#### --------------------------------------------------------------------------------------------------------------------------------------------------------------
# Solve for trajectories
#### --------------------------------------------------------------------------------------------------------------------------------------------------------------
sol_ECI_C = solve_ivp(Orbital_DiffyQ, t_span, ICs_ECI_C, t_eval=ts, method='RK45',rtol=1e-10) # Solve the system of differential equations
sol_ECI_T = solve_ivp(Orbital_DiffyQ, t_span, ICs_ECI_T, t_eval=ts, method='RK45',rtol=1e-10) # Solve the system of differential equations


r_ECI_C = np.array([sol_ECI_C.y[0], sol_ECI_C.y[1], sol_ECI_C.y[2]])
v_ECI_C = np.array([sol_ECI_C.y[3], sol_ECI_C.y[4], sol_ECI_C.y[5]])
r_ECI_T = np.array([sol_ECI_T.y[0], sol_ECI_T.y[1], sol_ECI_T.y[2]])
v_ECI_T = np.array([sol_ECI_T.y[3], sol_ECI_T.y[4], sol_ECI_T.y[5]])

print(r_ECI_C.dtype)

print(np.array([sol_ECI_C.t])[0,:])
rvt_T_ECI = np.array([r_ECI_T[0,:].T, r_ECI_T[1,:].T, r_ECI_T[2,:].T, v_ECI_T[0,:].T, v_ECI_T[1,:].T, v_ECI_T[2,:].T, np.array([sol_ECI_T.t])[0,:]]) # [rx ry rz vx vy vz t], treat this as a global variable for now
print(rvt_T_ECI)
print(rvt_T_ECI[6,:])
print(np.where(rvt_T_ECI[6,:] == 3)[0])
print(rvt_T_ECI[3:6,np.where(rvt_T_ECI[6,:] == 2)[0]])

test = np.array([rvt_T_ECI[3,np.where(rvt_T_ECI[6,:] == 2)[0]][0], rvt_T_ECI[4,np.where(rvt_T_ECI[6,:] == 2)[0]][0] ])
test2 = np.array([1,2,3])
print(test)
print(test.shape)
print(test2)
print(test2.shape)


sol_LVLH = solve_ivp(Hill_eqns, t_span, ICs_LVLH_C, t_eval=ts, method='RK45',rtol=1e-10) # Solve the system of differential equations
r_LVLH_C = np.array([sol_LVLH.y[0], sol_LVLH.y[1], sol_LVLH.y[2]])
v_LVLH_C = np.array([sol_LVLH.y[3], sol_LVLH.y[4], sol_LVLH.y[5]])
t_LVLH = sol_LVLH.t


sol_COMB_C = solve_ivp(Combined_DiffyQ, t_span, ICs_COMB_C, t_eval=ts, method='RK45',rtol=1e-10)
r_ECICOMB_T = np.array([sol_COMB_C.y[0], sol_COMB_C.y[1], sol_COMB_C.y[2]])
v_ECICOMB_T = np.array([sol_COMB_C.y[3], sol_COMB_C.y[4], sol_COMB_C.y[5]])
r_ECICOMB_C = np.array([sol_COMB_C.y[6], sol_COMB_C.y[7], sol_COMB_C.y[8]])
v_ECICOMB_C = np.array([sol_COMB_C.y[9], sol_COMB_C.y[10], sol_COMB_C.y[11]])
r_LVLHCOMB_C = np.array([sol_COMB_C.y[12], sol_COMB_C.y[13], sol_COMB_C.y[14]])
v_LVLHCOMB_C = np.array([sol_COMB_C.y[15], sol_COMB_C.y[16], sol_COMB_C.y[17]])

#r_LVLH_CONV
# print(r_ECICOMB_C)
# print(r_ECI_C)
# print(r_ECICOMB_C - r_ECI_C)
# print(r_LVLH_C)
# print(t_LVLH)

# print(len(t_LVLH))
# print(r_LVLH_C.shape[1])

r_CONV_C = np.zeros(r_LVLH_C.shape) # this is for LVLH converted to ECI
v_CONV_C = np.zeros(r_LVLH_C.shape)

r_error =  np.zeros(r_LVLH_C.shape)
v_error = np.zeros(r_LVLH_C.shape)

for ii in range(0,r_LVLH_C.shape[1]):
    rii, vii = LVLH2ECI(r_ECI_T[:,ii], v_ECI_T[:,ii], r_LVLH_C[:,ii], v_LVLH_C[:,ii])
    rii = rii/1000 + r_ECI_T[:,ii]
    vii = vii/1000 + v_ECI_T[:,ii]
    r_CONV_C[:,ii] = rii
    v_CONV_C[:,ii] = vii

    x = 1000*(rii[0]-r_ECI_C[0,ii])
    y = 1000*(rii[1]-r_ECI_C[1,ii])
    z = 1000*(rii[2]-r_ECI_C[2,ii])
    xdot = 1000*(vii[0]-v_ECI_C[0,ii])
    ydot = 1000*(vii[1]-v_ECI_C[1,ii])
    zdot = 1000*(vii[2]-v_ECI_C[2,ii])
    
    r_error[:,ii] = x,y,z
    v_error[:,ii] = xdot,ydot,zdot
    
r_COMBCONV_C = np.zeros(r_LVLHCOMB_C.shape) # this is for LVLH converted to ECI
v_COMBCONV_C = np.zeros(r_LVLHCOMB_C.shape)
r_errorCOMB =  np.zeros(r_LVLHCOMB_C.shape)
v_errorCOMB = np.zeros(r_LVLHCOMB_C.shape)
for ii in range(0,r_LVLHCOMB_C.shape[1]):
    rii, vii = LVLH2ECI(r_ECICOMB_T[:,ii], v_ECICOMB_T[:,ii], r_LVLHCOMB_C[:,ii], v_LVLHCOMB_C[:,ii])
    rii = rii/1000 + r_ECICOMB_T[:,ii] # m --> Km
    vii = vii/1000 + v_ECICOMB_T[:,ii]
    r_COMBCONV_C[:,ii] = rii
    v_COMBCONV_C[:,ii] = vii

    x = 1000*(rii[0]-r_ECICOMB_C[0,ii])
    y = 1000*(rii[1]-r_ECICOMB_C[1,ii])
    z = 1000*(rii[2]-r_ECICOMB_C[2,ii])
    xdot = 1000*(vii[0]-v_ECICOMB_C[0,ii])
    ydot = 1000*(vii[1]-v_ECICOMB_C[1,ii])
    zdot = 1000*(vii[2]-v_ECICOMB_C[2,ii])
    
    r_errorCOMB[:,ii] = x,y,z
    v_errorCOMB[:,ii] = xdot,ydot,zdot
#### --------------------------------------------------------------------------------------------------------------------------------------------------------------
# Plotting
#### --------------------------------------------------------------------------------------------------------------------------------------------------------------

# plot in lvlh:
fig = plt.figure(figsize=(8, 4))
ax1 = fig.add_subplot(1,2,1)
ax1.scatter(0, 0, s=100, marker='.', c='k', label='origin')
ax1.plot(r_LVLH_C[0], r_LVLH_C[1], c='b')

ax1.set_xlabel('x (r-bar), m')
ax1.set_ylabel('y (v-bar), m')

# ax1 = fig.add_subplot(1,2,1, projection='3d')
# ax1.scatter(0, 0, 0, s=100, marker='.', c='k', label="origin")

# plottraj(sol_lvlh.y[0], sol_lvlh.y[1], sol_lvlh.y[2], "lvlh", 'b', ax1)

# ax1.set_xlabel('x, m')
# ax1.set_ylabel('y, m')
# ax1.set_zlabel('z, m')
# ax1.tick_params(axis='x', labelsize=10)
# ax1.tick_params(axis='y', labelsize=10)
# ax1.tick_params(axis='z', labelsize=10)

# ax1.set_xlim([-np.max(abs(r_lvlh_c[0,:])), np.max(abs(r_lvlh_c[0,:]))])
# ax1.set_ylim([-np.max(abs(r_lvlh_c)), np.max(abs(r_lvlh_c))])
# ax1.set_zlim([-np.max(abs(r_lvlh_c)), np.max(abs(r_lvlh_c))])
ax1.set_title("lvlh frame, {} orbits".format(round(t/tau,2)))
plt.grid(True)
ax1.legend()

#plot trajectory in eci:
ax2 = fig.add_subplot(1,2,2, projection='3d')
plotTraj(r_ECI_T[0], r_ECI_T[1], r_ECI_T[2], "target", 'r', ax2, Marker=True)
plotTraj(r_ECI_C[0], r_ECI_C[1], r_ECI_C[2], "chaser", 'g', ax2, Marker=True)
plotTraj(r_CONV_C[0], r_CONV_C[1], r_CONV_C[2], "Converted", 'b', ax2, Marker=True)

ax2.scatter(0, 0, 0, s=100, marker='.', c='k', label="origin")

ax2.set_xlabel('x, km')
ax2.set_ylabel('y, km')
ax2.set_zlabel('z, km')
ax2.tick_params(axis='x', labelsize=10)
ax2.tick_params(axis='y', labelsize=10)
ax2.tick_params(axis='z', labelsize=10)

ax2.set_title("eci, {} orbits".format(round(t/tau,2)))
ax2.legend()
#plt.show()

fig = plt.figure(figsize=(16, 10))
plt.title("Comparision where no forces are applied to chaser")
ax1 = fig.add_subplot(2,3,1)
ax1.plot(t_LVLH, r_error[0,: ])
ax1.set_xlabel("time, sec")
ax1.set_ylabel("x error, m")
plt.grid(True)
ax1 = fig.add_subplot(2,3,2)
ax1.plot(t_LVLH, r_error[1,: ])
ax1.set_xlabel("time, sec")
ax1.set_ylabel("y error, m")
plt.grid(True)
ax1 = fig.add_subplot(2,3,3)
ax1.plot(t_LVLH, r_error[2,: ])
ax1.set_xlabel("time, sec")
ax1.set_ylabel("z error, m")
plt.grid(True)
ax1 = fig.add_subplot(2,3,4)
ax1.plot(t_LVLH, v_error[0,: ])
ax1.set_xlabel("time, sec")
ax1.set_ylabel("xdot error, m/sec")
plt.grid(True)
ax1 = fig.add_subplot(2,3,5)
ax1.plot(t_LVLH, v_error[1,: ])
ax1.set_xlabel("time, sec")
ax1.set_ylabel("ydot error, m/sec")
plt.grid(True)
ax1 = fig.add_subplot(2,3,6)
ax1.plot(t_LVLH, v_error[2,: ])
ax1.set_xlabel("time, sec")
ax1.set_ylabel("zdot error, m/sec")
plt.grid(True)
plt.show()





######### TESTING THE J2 PERTURBATION AGAINST TEXTBOOK

# r_test = np.array([-2384.46, 5729.01, 3050.46])
# v_test = np.array([-7.36138, -2.98997, 1.64354])

# tf = int(48*60*60)
# ts = np.linspace(0,tf,tf+1)
# t_span = (0, tf+1)

# ICs_test = [r_test[0], r_test[1], r_test[2], v_test[0], v_test[1], v_test[2]]
# sol_test = solve_ivp(Orbital_DiffyQ, t_span, ICs_test, t_eval=ts, method='RK45',rtol=1e-10) # Solve the system of differential equations

# r_testsol = np.array([sol_test.y[0], sol_test.y[1], sol_test.y[2]])
# v_testsol = np.array([sol_test.y[3], sol_test.y[4], sol_test.y[5]])


# print(sol_test.t)
# coes = np.zeros([6,r_testsol.shape[1]])

# dw = np.zeros([1,r_testsol.shape[1]])
# dRA = np.zeros([1,r_testsol.shape[1]])

# w0 = np.degrees(coe_from_sv(r_testsol[:,0], v_testsol[:,0], mu)[3])
# RA0 = np.degrees(coe_from_sv(r_testsol[:,0], v_testsol[:,0], mu)[1])
# print(dw[0,:])


# for ii in range(0,r_testsol.shape[1]):
#     coes[:,ii] = coe_from_sv(r_testsol[:,ii], v_testsol[:,ii], mu)

#     dw[0,ii] = np.degrees(coes[3,ii]) - w0
#     dRA[0,ii] = np.degrees(coes[1,ii]) - RA0
    
# fig = plt.figure(figsize=(8,4))
# ax3 = fig.add_subplot(1,2,1)
# ax3.plot(sol_test.t/3600,dw[0,:])
# plt.title("dw")
# plt.grid(True)

# print(dw.shape)
# print(sol_test.t[1].shape)
# ax3 = fig.add_subplot(2,2,2)
# ax3.plot(sol_test.t/3600, dRA[0,:])
# plt.title("dRA")
# plt.grid(True)

# plt.show()


# fig = plt.figure(figsize=(16, 10))
# ax1 = fig.add_subplot(1,3,1)
# ax1.plot(t_LVLH,r_ECICOMB_C[0]-r_ECI_C[0])
# ax1 = fig.add_subplot(2,3,2)
# ax1.plot(t_LVLH,r_ECICOMB_C[1]-r_ECI_C[1])
# ax1 = fig.add_subplot(3,3,3)
# ax1.plot(t_LVLH,r_ECICOMB_C[2]-r_ECI_C[2])
# plt.show()


x_rms = np.sqrt(np.mean(np.square(r_errorCOMB[0,:])))
y_rms = np.sqrt(np.mean(np.square(r_errorCOMB[1,:])))
z_rms = np.sqrt(np.mean(np.square(r_errorCOMB[2,:])))
xdot_rms = np.sqrt(np.mean(np.square(v_errorCOMB[0,:])))
ydot_rms = np.sqrt(np.mean(np.square(v_errorCOMB[1,:])))
zdot_rms = np.sqrt(np.mean(np.square(v_errorCOMB[2,:])))


fig, axs = plt.subplots(2, 3, figsize=(18,10))
ax1 = axs[0,0]
ax2 = axs[0,1]
ax3 = axs[0,2]
ax4 = axs[1,0]
ax5 = axs[1,1]
ax6 = axs[1,2]

# fig.suptitle('Comparision where forces ARE applied to chaser', y=0.95, fontsize=20)
fig.suptitle('Comparision for a forced motion trajectory', y=0.95, fontsize=20)

ax1.set_title('x_RMS: {}'.format(round_to_sig_figs(x_rms,3)))
ax1.plot(t_LVLH, r_errorCOMB[0,: ])
ax1.set_xlabel("time, sec")
ax1.set_ylabel("x error, m")
ax1.grid()

ax2.set_title('y_RMS: {}'.format(round_to_sig_figs(y_rms,3)))
ax2.plot(t_LVLH, r_errorCOMB[1,: ])
ax2.set_xlabel("time, sec")
ax2.set_ylabel("y error, m")
ax2.grid()

ax3.set_title('z_RMS: {}'.format(round_to_sig_figs(z_rms,3)))
ax3.plot(t_LVLH, r_errorCOMB[2,: ])
ax3.set_xlabel("time, sec")
ax3.set_ylabel("z error, m")
ax3.grid()

ax4.set_title('xdot_RMS: {}'.format(round_to_sig_figs(xdot_rms,3)))
ax4.plot(t_LVLH, v_errorCOMB[0,: ])
ax4.set_xlabel("time, sec")
ax4.set_ylabel("xdot error, m/sec")
ax4.grid()

ax5.set_title('ydot_RMS: {}'.format(round_to_sig_figs(ydot_rms,3)))
ax5.plot(t_LVLH, v_errorCOMB[1,: ])
ax5.set_xlabel("time, sec")
ax5.set_ylabel("ydot error, m/sec")
ax5.grid()

ax6.set_title('zdot_RMS: {}'.format(round_to_sig_figs(zdot_rms,3)))
ax6.plot(t_LVLH, v_errorCOMB[2,: ])
ax6.set_xlabel("time, sec")
ax6.set_ylabel("zdot error, m/sec")
ax6.grid()

plt.subplots_adjust(wspace=0.25, hspace=0.3)

plt.show()

