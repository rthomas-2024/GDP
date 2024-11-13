###############################################
#                  IMPORTS                   #
###############################################
import numpy as np
import matplotlib.pyplot as plt
import scipy as sc
from scipy.linalg import norm
import sys

from scipy.integrate import solve_ivp
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation



###############################################
#                 FUNCTIONS                   #
###############################################
# UTILITY FUNCTIONS (ATTITUDE)
def triangleInequality(I):
    #input I as the full inertial matrix
    I1 = I[0,0]
    I2 = I[1,1]
    I3 = I[2,2]

    if (I1+I2) < I3 or (I2+I3) < I1 or (I1+I3) < I2:
        print('Impossible object, does not satisfy triangle inequality')
        sys.exit()
def getAmat(omega):
    #this gets the matrix used for iteration of quaternions BEFORE the exponential function
    w1, w2, w3 = omega
    A = np.array([[0, -w1, -w2, -w3],
                 [w1, 0, w3, -w2],
                 [w2, -w3, 0, w1],
                 [w3, w2, -w1, 0]])

    return A
def inverseQuaternion(beta):
    b0, b1, b2, b3 = beta
    invBeta = np.array([b0, -b1, -b2, -b3])
    return invBeta
def vecToPureQuaternion(r):
    #returns a pure quaternion for quaternion multiplication

    #initialise the pure quaternion
    rDash = np.zeros([4])
    
    #basically just stick a zero in the scalar part (ofc bcause its a vector)
    rDash[1:4] = r

    return rDash
def pureQuaternionToVec(rDash):
    r = np.zeros([3])

    r = rDash[1:4]

    return r
def quaternionMult(q, b):
    #multiplies the quaternions q and b (in that order)

    q0, q1, q2, q3 = q

    qMat = np.array([[q0, -q1, -q2, -q3],
                     [q1, q0, -q3, q2],
                     [q2, q3, q0, -q1],
                     [q3, -q2, q1, q0]])

    return qMat @ b

#TRANSFORMATION FUNCTIONS (ATTITUDE)
def quaternionToDCM(beta):
    #beta passed in as a quaternion (4 value vector)
    b0, b1, b2, b3 = beta

    C11 = b0**2+b1**2-b2**2-b3**2
    C12 = 2*(b1*b2+b0*b3)
    C13 = 2*(b1*b3-b1*b2)
    C21 = 2*(b1*b2-b0*b3)
    C22 = b0**2-b1**2+b2**2-b3**2
    C23 = 2*(b2*b3+b0*b1)
    C31 = 2*(b1*b3+b0*b2)
    C32 = 2*(b2*b3-b0*b1)
    C33 = b0**2-b1**2-b2**2+b3**2

    C = np.array([[C11, C12, C13],
                 [C21, C22, C23],
                 [C31, C32, C33]])
    return C
def DCMtoQuaternion(dcm):
    # Ensure dcm is a 3x3 matrix
    assert dcm.shape == (3, 3)
    
    trace = np.trace(dcm)
    
    # Initialize quaternion
    q = np.zeros(4)
    
    # Compute quaternion based on the trace value
    if trace > 0:
        s = np.sqrt(trace + 1.0) * 2  # s = 4 * q_w
        q[0] = 0.25 * s
        q[1] = (dcm[2, 1] - dcm[1, 2]) / s
        q[2] = (dcm[0, 2] - dcm[2, 0]) / s
        q[3] = (dcm[1, 0] - dcm[0, 1]) / s
    elif (dcm[0, 0] > dcm[1, 1]) and (dcm[0, 0] > dcm[2, 2]):
        s = np.sqrt(1.0 + dcm[0, 0] - dcm[1, 1] - dcm[2, 2]) * 2  # s = 4 * q_x
        q[0] = (dcm[2, 1] - dcm[1, 2]) / s
        q[1] = 0.25 * s
        q[2] = (dcm[0, 1] + dcm[1, 0]) / s
        q[3] = (dcm[0, 2] + dcm[2, 0]) / s
    elif dcm[1, 1] > dcm[2, 2]:
        s = np.sqrt(1.0 + dcm[1, 1] - dcm[0, 0] - dcm[2, 2]) * 2  # s = 4 * q_y
        q[0] = (dcm[0, 2] - dcm[2, 0]) / s
        q[1] = (dcm[0, 1] + dcm[1, 0]) / s
        q[2] = 0.25 * s
        q[3] = (dcm[1, 2] + dcm[2, 1]) / s
    else:
        s = np.sqrt(1.0 + dcm[2, 2] - dcm[0, 0] - dcm[1, 1]) * 2  # s = 4 * q_z
        q[0] = (dcm[1, 0] - dcm[0, 1]) / s
        q[1] = (dcm[0, 2] + dcm[2, 0]) / s
        q[2] = (dcm[1, 2] + dcm[2, 1]) / s
        q[3] = 0.25 * s
    
    return q  #LOOK OVER AND POSSIBLY CHANGE
def eulerToDCM(rollPitchYaw):
    roll, pitch, yaw = rollPitchYaw
    # Compute individual rotation matrices
    R_x = np.array([
        [1, 0, 0],
        [0, np.cos(roll), -np.sin(roll)],
        [0, np.sin(roll), np.cos(roll)]
    ])
    
    R_y = np.array([
        [np.cos(pitch), 0, np.sin(pitch)],
        [0, 1, 0],
        [-np.sin(pitch), 0, np.cos(pitch)]
    ])
    
    R_z = np.array([
        [np.cos(yaw), -np.sin(yaw), 0],
        [np.sin(yaw), np.cos(yaw), 0],
        [0, 0, 1]
    ])
    
    # Combined rotation matrix for ZYX order (yaw -> pitch -> roll)
    dcm = R_z @ R_y @ R_x
    return dcm #LOOK OVER AND POSSIBLY CHANGE
def DCMtoEuler(dcm):

    if abs(dcm[2, 0]) != 1:
        # Calculate yaw, pitch, roll angles
        pitch = -np.arcsin(dcm[2, 0])
        roll = np.arctan2(dcm[2, 1] / np.cos(pitch), dcm[2, 2] / np.cos(pitch))
        yaw = np.arctan2(dcm[1, 0] / np.cos(pitch), dcm[0, 0] / np.cos(pitch))
    else:
        #Gimbal lock occurs, choose yaw = 0
        yaw = 0
        if dcm[2, 0] == -1:
            pitch = np.pi / 2
            roll = yaw + np.arctan2(dcm[0, 1], dcm[0, 2])
        else:
            pitch = -np.pi / 2
            roll = -yaw + np.arctan2(-dcm[0, 1], -dcm[0, 2])

    return np.array([roll,pitch,yaw]) #LOOK OVER AND POSSIBLY CHANGE

#CORE FUNCTIONS (ATTITUDE)
def EulerEquations(t, stateVec, T_ext_func):
    #I is the full inertial matrix, and omega is an angular velocity vector
    I11 = InertMat[0,0]
    I22 = InertMat[1,1]
    I33 = InertMat[2,2]

    omega = stateVec[0:3]
    q = stateVec[3:7]

    omega1, omega2, omega3 = omega
    T1, T2, T3 = T_ext_func(t)

    dw1dt = (T1 - (I33-I22)*omega2*omega3) / I11
    dw2dt = (T2 - (I11-I33)*omega1*omega3) / I22
    dw3dt = (T3 - (I22-I11)*omega2*omega1) / I33

    omegaDot = np.array([dw1dt, dw2dt, dw3dt]) #returns the dw/dt full vector
    qDot = getAmat(omega) @ q

    stateVecDot = np.zeros([7])
    stateVecDot[0:3] = omegaDot
    stateVecDot[3:7] = qDot

    #note quaternions used because it creates smooth interpolation for animations. this is called slerp

    return stateVecDot
def getVertices(centroid, length, q):
    #returns vertices with centroid and length arguments
    vertArr = np.array([[centroid[0]-0.5*length, centroid[1]-0.5*length, centroid[2]-0.5*length],
                        [centroid[0]+0.5*length, centroid[1]-0.5*length, centroid[2]-0.5*length],
                        [centroid[0]+0.5*length, centroid[1]-0.5*length, centroid[2]+0.5*length],
                        [centroid[0]-0.5*length, centroid[1]-0.5*length, centroid[2]+0.5*length],
                        [centroid[0]-0.5*length, centroid[1]+0.5*length, centroid[2]-0.5*length],
                        [centroid[0]+0.5*length, centroid[1]+0.5*length, centroid[2]-0.5*length],
                        [centroid[0]+0.5*length, centroid[1]+0.5*length, centroid[2]+0.5*length],
                        [centroid[0]-0.5*length, centroid[1]+0.5*length, centroid[2]+0.5*length]])

    #'move' centroid to global origin to ensure rotation carried out correctly
    vertArr -= centroid

    for i in range(8):
        verti = vecToPureQuaternion(vertArr[i,:])
        verti = quaternionMult(q, quaternionMult(verti, inverseQuaternion(q)))
        verti = pureQuaternionToVec(verti)
        vertArr[i,:] = verti

    #'move' centroid back to correct position
    vertArr += centroid

    return vertArr
def plotCube(vertices):
    #plots 'front and back' faces
    for i in range(3):
        xvals = np.array([vertices[i, 0], vertices[i+1, 0]])
        yvals = np.array([vertices[i, 1], vertices[i+1, 1]])
        zvals = np.array([vertices[i, 2], vertices[i+1, 2]])
        ax.plot3D(xvals, yvals, zvals, color = 'b')

    xvals = np.array([vertices[3, 0], vertices[0, 0]])
    yvals = np.array([vertices[3, 1], vertices[0, 1]])
    zvals = np.array([vertices[3, 2], vertices[0, 2]])
    ax.plot3D(xvals, yvals, zvals, color = 'b')

    for i in range(3):
        xvals = np.array([vertices[i+4, 0], vertices[i+5, 0]])
        yvals = np.array([vertices[i+4, 1], vertices[i+5, 1]])
        zvals = np.array([vertices[i+4, 2], vertices[i+5, 2]])
        ax.plot3D(xvals, yvals, zvals, color = 'b')

    xvals = np.array([vertices[7, 0], vertices[4, 0]])
    yvals = np.array([vertices[7, 1], vertices[4, 1]])
    zvals = np.array([vertices[7, 2], vertices[4, 2]])
    ax.plot3D(xvals, yvals, zvals, color = 'b')

    #plots lines connecting them
    for i in range(4):
        xvals = np.array([vertices[i, 0], vertices[i+4, 0]])
        yvals = np.array([vertices[i, 1], vertices[i+4, 1]])
        zvals = np.array([vertices[i, 2], vertices[i+4, 2]])
        ax.plot3D(xvals, yvals, zvals, color = 'r')
def getAttitudes(theta0, ts, qs):
    thetas = np.zeros([len(ts), 3])
    thetas[0,:] = theta0

    for i in range(len(ts)-1):
        thetas[i+1,:] = DCMtoEuler(quaternionToDCM(qs[i+1,:]))

    return thetas

#TRAJECTORY FUNCTIONS
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




###############################################
#                   INPUTS                    #
###############################################
InertMat = np.array([[1,0,0], [0,1,0],[0,0,1]]) #inertial matrix
w0 = np.array([0.3,1,0]) #initial angular velocity
theta0 = np.array([30,15,10]) #initial attitude in degrees (roll, pitch, yaw)

def T_ext_func(t): #define the thrust over time in body frame
   T1 = 0
   T2 = 0
   T3 = 0
   return np.array([T1, T2, T3])

tspan = np.array([0, 120]) #spans one minute (start and stop)
dt = 0.01 #timestep in seconds

triangleInequality(InertMat) #checks that the object exists
theta0 = theta0 * 2*np.pi/360 #convert attitude to radians

#TRAJECTORY INPUTS
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

# Straight line V-bar approach with constant velocity
x0 = 0
y0 = 0
z0 = 0
dx0 = 0
dy0 = 0.01
dz0 = 0
f_x = lambda t: -2*n*dy0  # Define external force as a function of time

ICs_LVLH_C = [x0, y0, z0, dx0, dy0, dz0]  # [x0, y0, z0, dx0, dy0, dz0], initial conditions
ICs_ECI_T = [rT[0], rT[1], rT[2], vT[0], vT[1], vT[2]]



###############################################
# FIND ANGULAR VELO AND QUATERNIONS OVER TIME #
###############################################
t_eval = np.arange(tspan[0], tspan[1]+dt, dt) #when to store state matrix
q0 = DCMtoQuaternion(eulerToDCM(theta0)) #get intial quaternion

#initialise the initial state vector (made up of angular velos and quaternions at each timestep)
isv = np.zeros([7])

#fill initial state vector
isv[0:3] = w0
isv[3:7] = q0

fullSolution = sc.integrate.solve_ivp(EulerEquations, tspan, isv, t_eval = t_eval, args=(T_ext_func,))

#called it full solution because it contains lots of useless information
#we just want how state vector changes over time

omegaVec = fullSolution.y[0:3, :] #the .y exctracts just the omegas over the tspan
qs = fullSolution.y[3:7, :]

#find the error in the norm of the quaternions from 1
qErr = np.zeros([len(qs[0,:])])

for i in range(len(qs[0,:])):
    qErr[i] = 1 - norm(qs[:,i])


#find the thrust values over time
T1s = [T_ext_func(t)[0] for t in t_eval]
T2s = [T_ext_func(t)[1] for t in t_eval]
T3s = [T_ext_func(t)[2] for t in t_eval]

#TRAJECORY SOLVING
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



#PLOTTING
fig1, axs = plt.subplots(2, 2, figsize=(15,10))
ax1 = axs[0,1] #w
ax2 = axs[1,0] #q
ax3 = axs[1,1] #qErr
ax4 = axs[0,0] #T

#plot angular velocities
ax1.set_title('Angular Velocity Variation (Body Frame)')
ax1.plot(t_eval, omegaVec[0], color = 'b', label='omega_x')
ax1.plot(t_eval, omegaVec[1], color = 'r', label='omega_y')
ax1.plot(t_eval, omegaVec[2], color = 'g', label='omega_z')
ax1.grid()
ax1.set_xlabel('time (s)')
ax1.set_ylabel('angular velocity (rad/s)')
ax1.legend()

#plot quaternions
ax2.set_title('Quaternion Variation')
ax2.plot(t_eval, qs[0,:], color='b', label='q0')
ax2.plot(t_eval, qs[1,:], color='r', label='q1')
ax2.plot(t_eval, qs[2,:], color='g', label='q2')
ax2.plot(t_eval, qs[3,:], color='m', label='q3')
ax2.grid()
ax2.set_xlabel('time (s)')
ax2.set_ylabel('Quaternions')
ax2.legend()

#plot quaternion error (absolute)
ax3.set_title('Quaternion Error Variation (Absolute)')
ax3.plot(t_eval, qErr)
ax3.grid()
ax3.set_xlabel("Time (s)")
ax3.set_ylabel("Quaternion Norm Error (Absolute)")

#plot thrust
ax4.set_title('Thrust Variation')
ax4.plot(t_eval, T1s, color='b', label='T1')
ax4.plot(t_eval, T2s, color='r', label='T2')
ax4.plot(t_eval, T3s, color='g', label='T3')
ax4.grid()
ax4.set_xlabel('Time (s)')
ax4.set_ylabel('Thrust (N)')
ax4.legend()

plt.subplots_adjust(wspace=0.25, hspace=0.3)
plt.show()



###############################################
#               CUBE PLOTTING                 #
###############################################
# fig2 = plt.figure(figsize = (10, 10))
# ax = plt.axes(projection = '3d')

# centroid = np.array([0.5,0.5,0.5])
# length = 1

# for i in range(len(t_eval)):
#     ax.clear()
#     ax.set_xlim(-2, 2)
#     ax.set_ylim(-2, 2)
#     ax.set_zlim(-2, 2)
#     ax.set_xlabel('x')
#     ax.set_xlabel('y')
#     ax.set_zlabel('z')
#     ax.set_title('Cube Plot')
#     ax.set_aspect('equal')

#     vert = getVertices(centroid, length, qs[:,i])
#     plotCube(vert)
#     ax.plot3D(centroid[0], centroid[1], centroid[2], marker=".", markersize=10, color="g")

#     plt.pause(dt)

# plt.show()


#### --------------------------------------------------------------------------------------------------------------------------------------------------------------
#  TRAJECTORY Plotting
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

# Animate both plots
data = {
    'LVLH': (sol_LVLH.y[0], sol_LVLH.y[1], sol_LVLH.y[2]),
    'ECI': (r_ECI_C[0], r_ECI_C[1], r_ECI_C[2], sol_ECI_T.y[0], sol_ECI_T.y[1], sol_ECI_T.y[2])
}
animate_3d_trajectories(data, framerate=200, ANIMATE=False)
