###############################################
#                  IMPORTS                    #
###############################################
from math import degrees, radians
import numpy as np
from numpy import sin as s
from numpy import cos as c
from numpy import pi as pi
import matplotlib.pyplot as plt
import scipy as sc
from scipy import interpolate
from scipy.linalg import norm
import sys
import time
import pybullet as p
from scipy.spatial.transform import Rotation as R

from scipy.integrate import solve_ivp
from matplotlib.animation import FuncAnimation
from scipy.interpolate import interp1d
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

import pandas as pd


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
def quatToVisQuat(q):
    #moves the scalar part of the quaternion from the start to the end of the quaternion as this is how it is defined in pybullet
    q0,q1,q2,q3 = q
    
    q[0] = q1
    q[1] = q2
    q[2] = q3
    q[3] = q0

    return q
def getCosPitch(C11, C12, yaw):
    #get cos(pitch) from C11 or C12 because yaw is known
    #cannot find if the DCM component is 0
    #but will never both be zero unless in gimbal lock
    if C11 > 1e-10 or C11 < -1e-10:
        #essentially is C11 is not equal to 0, but have to acount for numerical errors, wont be exactly 0
        cosPitch = C11/c(yaw)
    else:
        #this is the case where C11 IS equal to 0, must use C12
        cosPitch = C12/s(yaw)

    return cosPitch
def getGimbalLockAngles(sinPitch, C22, C32):
    print('GIMBAL LOCK')
    #can of course directlty find pitch
    if sinPitch == 1:
        pitch = pi/2
    elif sinPitch == -1:
        pitch = -pi/2
    else:
        print("gimbal lock error 1")

    #use the 'trick' of arbitrarily setting yaw to 0
    yaw = 0

    #now find roll (see derivation of this on document)
    roll = np.arctan2(-C32, C22)

    return roll, pitch, yaw

#TRANSFORMATION FUNCTIONS (ATTITUDE)
def quaternionToDCM(beta):
    #beta passed in as a quaternion (4 value vector)
    b0, b1, b2, b3 = beta

    C11 = b0**2+b1**2-b2**2-b3**2
    C12 = 2*(b1*b2+b0*b3)
    C13 = 2*(b1*b3-b0*b2)
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
    
    C11,C12,C13 = dcm[0,0:3]
    C21,C22,C23 = dcm[1,0:3]
    C31,C32,C33 = dcm[2,0:3]    

    b0 = 0.5*np.sqrt(trace+1)

    if b0!=0:
        b1 = (C23-C32)/(4*b0)
        b2 = (C31-C13)/(4*b0)
        b3 = (C12-C21)/(4*b0)
    else: #if singularity, use sheppards algorithm
        b02 = 0.25*(1+trace)
        b12 = 0.25*(1+2*C11-trace)
        b22 = 0.25*(1+2*C22-trace)
        b32 = 0.25*(1+2*C33-trace)

        bs2 = np.array([b02, b12, b22, b32])

        if b02 == bs2.max():
            b0 = np.sqrt(b02)
            b1 = (C23-C32)/(4*b0)
            b2 = (C31-C13)/(4*b0)
            b3 = (C12-C21)/(4*b0)
        elif b12 == bs2.max():
            b1 = np.sqrt(b12)
            b0 = (C23-C32)/(4*b1)
            b2 = (C12+C21)/(4*b1)
            b3 = (C31+C13)/(4*b1)
        elif b22 == bs2.max():
            b2 = np.sqrt(b22)
            b0 = (C31-C13)/(4*b2)
            b1 = (C12+C21)/(4*b2)
            b3 = (C23+C32)/(4*b2)
        elif b32 == bs2.max():
            b3 = np.sqrt(b32)
            b0 = (C12-C21)/(4*b3)
            b1 = (C31+C13)/(4*b3)
            b2 = (C23+C32)/(4*b3)

    q = np.array([b0,b1,b2,b3])
    return q
def eulerToDCM(rollPitchYaw):
    #we want to find the DCM that transforms an inertial vector to a body one
    #these are of course the 3-2-1 
    roll, pitch, yaw = rollPitchYaw

    #first rotation about the 3-axis (z-axis) is yaw
    R3 = np.array([[c(yaw), s(yaw), 0],
                  [-s(yaw), c(yaw), 0],
                   [0,0,1]])

    #second rotation about the 2-axis (y-axis) is pitch
    R2 = np.array([[c(pitch), 0, -s(pitch)],
                   [0,1,0],
                   [s(pitch), 0, c(pitch)]])

    #third rotation about the 1-axis (x-axis) is roll
    R1 = np.array([[1,0,0],
                   [0,c(roll),s(roll)],
                   [0, -s(roll), c(roll)]])

    #apply to inertial vec in order 3-2-1 (3 first so on the right, etc)
    dcm = R1 @ R2 @ R3

    return dcm
def DCMtoEuler(dcm):
    #this is again assuming that the input DCM is the DCM FROM inertial TO body
    C11, C12, C13 = dcm[0,:]
    C21, C22, C23 = dcm[1,:]
    C31, C32, C33 = dcm[2,:]

    #arctan2 does an automatic quadrant check
    yaw = np.arctan2(C12,C11)
    roll = np.arctan2(C23,C33)

    #now solve for pitch
    #we need sin(pitch) and cos(pitch) to determine the correct pitch angle
    sinPitch = -C13
    cosPitch = getCosPitch(C11,C12,yaw)

    if sinPitch == 1 or sinPitch == -1:
        #must check for gimbal lock FIRST of course, if found true, wont run the rest of the if
        roll, pitch, yaw = getGimbalLockAngles(sinPitch, C22, C32)

    elif sinPitch == 0:
        #first case if sin(pitch) is 0, check which value it takes depending on cos(pitch)
        if cosPitch > 0.99 and cosPitch < 1.01:
            #the above is essentially cosPitch == 1, but need to take into account numerical errors
            #numerical errors here dont have to be fine because if sinPitch is 0, cosPitch will be either 1 or -1, just have to be near
            pitch = 0
        elif cosPitch == cosPitch > -1.01 and cosPitch < -0.99:
            #same but for cosPitch == -1, taking into account numerical errors
            pitch = pi
        else:
            print("DCM to Euler error 1")

    elif sinPitch > 0:
        #second case if sin(pitch) is greater than 0, again check using cos(pitch)
        if cosPitch > 0:
            pitch = np.arcsin(sinPitch)
        elif cosPitch < 0:
            pitch = pi - np.arcsin(sinPitch)
        else:
            print("DCM to Euler error 2")

    elif sinPitch < 0:
        #third case if sin(pitch) is less than 0, again check using cos(pitch)
        if cosPitch > 0:
            pitch = np.arcsin(sinPitch)
        elif cosPitch < 0:
            pitch = -pi + abs(np.arcsin(sinPitch))
        else:
            print("DCM to Euler error 3")
    else:
        print("DCM to Euler error 4")

    
    return np.array([roll,pitch,yaw])

#CORE FUNCTIONS (ATTITUDE)
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
def pybulletPlot(centroidVec, q, dt, acc):
    p.resetBasePositionAndOrientation(chaser, centroidVec, quatToVisQuat(q))
    time.sleep(dt/acc)

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

# TRAJECTORY PLANNING FUNCTIONS
def cw_calc_dv0(dr0, dr1, t, n):
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
        if ii == NumWPs-1: # last waypoint
            t_ii = t_ii + dt # to include the very final index
        for t in np.arange(0,round(t_ii,6),dt): # this goes up to t = t_ii - dt
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
    ax.add_collection3d(Poly3DCollection(faces, alpha=0.2, edgecolor='k',label="Approach Corridor"))

    # Plot points
    RED_LEGEND = 1
    for x, y, z, inside in results:
        if inside == False and RED_LEGEND==1:
            ax.scatter(x, y, z, '.', c='r', s=5, label = "Outside Approach Corridor")
            RED_LEGEND = 0
        elif inside == False:
            ax.scatter(x, y, z, '.', c='r', s=5)

    ax.legend()
    # fig = plt.figure(figsize=(8, 4))
    # # ---- XY Projection ----
    # ax2 = fig.add_subplot(122)
    # xy_xs, xy_ys = zip(*xy_triangle)
    # ax2.fill(xy_xs, xy_ys, "b", alpha=0.5)

    # for x, y, _, inside in results:
    #     ax2.scatter(x, y, color='g' if inside else 'r', s=5)

    # ax2.set_xlabel("X")
    # ax2.set_ylabel("Y")
    # ax2.set_title("XY Projection")
    # ax2.set_aspect("equal")

    # # ---- ZX Projection ----
    # ax3 = fig.add_subplot(222)
    # zx_xs, zx_zs = zip(*zx_triangle)
    # ax3.fill(zx_xs, zx_zs, "r", alpha=0.5)

    # for x, _, z, inside in results:
    #     ax3.scatter(x, z, color='g' if inside else 'r', s=5)

    # ax3.set_xlabel("X")
    # ax3.set_ylabel("Z")
    # ax3.set_title("ZX Projection")
    # ax3.set_aspect("equal")
def interpolate_3d(interp_dx, interp_dy, interp_dz, t_query):
    """
    Interpolates the 3D coordinates at a given time t_query.

    """

    # Interpolate at t_query
    x_t = interp_dx(t_query)
    y_t = interp_dy(t_query)
    z_t = interp_dz(t_query)

    return np.array([x_t, y_t, z_t])

def pid_control(t, error, Kp, Ki, Kd, integral,previous_error, previous_time):
    
    # Calculate time step dynamically
    dt = t - previous_time
    
    integral += error * dt
    derivative = (error - previous_error) / dt if dt > 0 else 0
    previous_error = error
    previous_time = t
    
    return Kp * error + Ki * integral + Kd * derivative, integral, previous_error, previous_time

# COMBINED DIFFERENTIAL EQUATION
def TrajandAtt(t,stateVec,T_ext_func,interp_dx,interp_dy,interp_dz):
    #I is the full inertial matrix, and omega is an angular velocity vector
    I11 = InertMat[0,0]
    I22 = InertMat[1,1]
    I33 = InertMat[2,2]

    omega = stateVec[0:3]
    q = stateVec[3:7]

    # Attitude control
    global prev_time, integral_x,integral_y,integral_z,prev_error_x,prev_error_y,prev_error_z, integral_roll,integral_pitch,integral_yaw,prev_error_roll,prev_error_pitch,prev_error_yaw
    prev_time_iter = prev_time
    C = quaternionToDCM(q)
    roll, pitch, yaw = DCMtoEuler(C)
    roll_err = roll_ref - roll
    pitch_err = pitch_ref - pitch
    yaw_err = yaw_ref - yaw
    q_err = 1-norm(q)
    # C_err = quaternionToDCM(q_err)
    # roll_err, pitch_err, yaw_err = DCMtoEuler(C_err)
    print("Quaternion error: {}".format(q_err))
    print("roll: {}".format(roll), "pitch: {}".format(pitch), "yaw: {}".format(yaw))
    u_roll, integral_roll, prev_error_roll, prev_time = pid_control(t, roll_err, kP_roll, kI_roll, kD_roll, integral_roll, prev_error_roll, prev_time_iter)
    u_pitch, integral_pitch, prev_error_pitch, prev_time = pid_control(t, pitch_err, kP_pitch, kI_pitch, kD_pitch, integral_pitch, prev_error_pitch, prev_time_iter)
    u_yaw, integral_yaw, prev_error_yaw, prev_time = pid_control(t, yaw_err, kP_yaw, kI_yaw, kD_yaw, integral_yaw, prev_error_yaw, prev_time_iter)

    print("u_roll: {}".format(u_roll),"u_pitch: {}".format(u_pitch),"u_yaw: {}".format(u_yaw))
    
    if u_roll > u_roll_thresh: u_roll = u_roll_max
    elif u_roll < -u_roll_thresh: u_roll = -u_roll_max
    else: u_roll = 0
    if u_pitch > u_pitch_thresh: u_pitch = u_pitch_max
    elif u_pitch < -u_pitch_thresh: u_pitch = -u_pitch_max
    else: u_pitch = 0
    if u_yaw > u_yaw_thresh: u_yaw = u_yaw_max
    elif u_yaw < -u_yaw_thresh: u_yaw = -u_yaw_max
    else: u_yaw = 0

    omega1, omega2, omega3 = omega
    #T1, T2, T3 = T_ext_func(t)
    T_control = np.dot(InertMat.T, np.array([u_roll,u_pitch,u_yaw]))
    dw1dt = (T_control[0] - (I33-I22)*omega2*omega3) / I11
    dw2dt = (T_control[1] - (I11-I33)*omega1*omega3) / I22
    dw3dt = (T_control[2] - (I22-I11)*omega2*omega1) / I33

    omegaDot = np.array([dw1dt, dw2dt, dw3dt]) #returns the dw/dt full vector
    qDot = getAmat(omega) @ q

    stateVecDot = np.zeros([25])
    stateVecDot[0:3] = omegaDot
    stateVecDot[3:7] = qDot

    #note quaternions used because it creates smooth interpolation for animations. this is called slerp

    ###########################
    # TRAJECTORY
    ###########################
    # unpack variables
    xT_ECI, yT_ECI, zT_ECI, dxT_ECI, dyT_ECI, dzT_ECI, x_ECI, y_ECI, z_ECI, dx_ECI, dy_ECI, dz_ECI, x_LVLH, y_LVLH, z_LVLH, dx_LVLH, dy_LVLH, dz_LVLH = stateVec[7:25]
    
    # 2 body acceleration, Target
    rT_ECI = np.array([xT_ECI, yT_ECI, zT_ECI])
    ddxT_ECI = (-mu/(np.linalg.norm(rT_ECI)**3)) * xT_ECI
    ddyT_ECI = (-mu/(np.linalg.norm(rT_ECI)**3)) * yT_ECI
    ddzT_ECI = (-mu/(np.linalg.norm(rT_ECI)**3)) * zT_ECI
    aT_ECI = np.array([ddxT_ECI, ddyT_ECI, ddzT_ECI])

    rT = rT_ECI
          
    # Trajectory Control
    dr = np.array([x_LVLH, y_LVLH, z_LVLH])
    dr_ref = interpolate_3d(interp_dx,interp_dy,interp_dz, t)
    dr_error = dr_ref - dr

    u_x, integral_x, prev_error_x, prev_time = pid_control(t, dr_error[0], kPx, kIx, kDx, integral_x, prev_error_x, prev_time_iter)
    u_y, integral_y, prev_error_y, prev_time = pid_control(t, dr_error[1], kPy, kIy, kDy, integral_y, prev_error_y, prev_time_iter)
    u_z, integral_z, prev_error_z, prev_time = pid_control(t, dr_error[2], kPz, kIz, kDz, integral_z, prev_error_z, prev_time_iter)
        
    #print( "input_x: {}".format(u_x))

    if u_x > u_x_thresh: u_x = u_x_max
    elif u_x < -u_x_thresh: u_x = -u_x_max
    else: u_x = 0
    if u_y > u_y_thresh: u_y = u_y_max
    elif u_y < -u_y_thresh: u_y = -u_y_max
    else: u_y = 0
    if u_z > u_z_thresh: u_z = u_z_max
    elif u_z < -u_z_thresh: u_z = -u_z_max
    else: u_z = 0

   # print("time: {}".format(t), "ref point: {}".format(dr_ref),"prev point: {}".format(dr) , "input_x: {}".format(u_x))
    #print( "input_x: {}".format(u_x))
    print(t)
    # Hill Eqns, Chaser
    ddx_LVLH = 2 * n * dy_LVLH + 3 * n**2 * x_LVLH  + u_x
    ddy_LVLH = -2 * n * dx_LVLH + u_y
    ddz_LVLH = -n**2 * z_LVLH + u_z
    a_LVLH = np.array([ddx_LVLH, ddy_LVLH, ddz_LVLH])

    # 2 body acceleration, Chaser
    r_ECI = np.array([x_ECI, y_ECI, z_ECI])
    ddx_ECI = (-mu/(np.linalg.norm(r_ECI)**3)) * x_ECI 
    ddy_ECI = (-mu/(np.linalg.norm(r_ECI)**3)) * y_ECI
    ddz_ECI = (-mu/(np.linalg.norm(r_ECI)**3)) * z_ECI
    a_ECI = np.array([ddx_ECI, ddy_ECI, ddz_ECI])
    
    stateVecDot[7:25] = dxT_ECI,dyT_ECI,dzT_ECI,aT_ECI[0],aT_ECI[1],aT_ECI[2], dx_ECI,dy_ECI,dz_ECI,a_ECI[0],a_ECI[1],a_ECI[2], dx_LVLH,dy_LVLH,dz_LVLH,a_LVLH[0],a_LVLH[1],a_LVLH[2]
        
    return stateVecDot
def TrajandAttREDUCED(t,stateVec,T_ext_func,interp_dx,interp_dy,interp_dz):
    #I is the full inertial matrix, and omega is an angular velocity vector
    I11 = InertMat[0,0]
    I22 = InertMat[1,1]
    I33 = InertMat[2,2]

    omega = stateVec[0:3]
    roll,pitch,yaw = stateVec[3:6]

    # Attitude control
    global prev_time, integral_x,integral_y,integral_z,prev_error_x,prev_error_y,prev_error_z, integral_roll,integral_pitch,integral_yaw,prev_error_roll,prev_error_pitch,prev_error_yaw
    prev_time_iter = prev_time

    roll_err = roll_ref - roll
    pitch_err = pitch_ref - pitch
    yaw_err = yaw_ref - yaw
    
    print("roll: {}".format(roll), "pitch: {}".format(pitch), "yaw: {}".format(yaw))
    u_roll, integral_roll, prev_error_roll, prev_time = pid_control(t, roll_err, kP_roll, kI_roll, kD_roll, integral_roll, prev_error_roll, prev_time_iter)
    u_pitch, integral_pitch, prev_error_pitch, prev_time = pid_control(t, pitch_err, kP_pitch, kI_pitch, kD_pitch, integral_pitch, prev_error_pitch, prev_time_iter)
    u_yaw, integral_yaw, prev_error_yaw, prev_time = pid_control(t, yaw_err, kP_yaw, kI_yaw, kD_yaw, integral_yaw, prev_error_yaw, prev_time_iter)
    #print("u_roll: {}".format(u_roll),"u_pitch: {}".format(u_pitch),"u_yaw: {}".format(u_yaw))
    
    if u_roll > u_roll_thresh: u_roll = u_roll_max
    elif u_roll < -u_roll_thresh: u_roll = -u_roll_max
    else: u_roll = 0
    if u_pitch > u_pitch_thresh: u_pitch = u_pitch_max
    elif u_pitch < -u_pitch_thresh: u_pitch = -u_pitch_max
    else: u_pitch = 0
    if u_yaw > u_yaw_thresh: u_yaw = u_yaw_max
    elif u_yaw < -u_yaw_thresh: u_yaw = -u_yaw_max
    else: u_yaw = 0

    omega1, omega2, omega3 = omega
    #T1, T2, T3 = T_ext_func(t)
    T_control = np.dot(InertMat.T, np.array([u_roll,u_pitch,u_yaw]))
    dw1dt = (T_control[0] - (I33-I22)*omega2*omega3) / I11
    dw2dt = (T_control[1] - (I11-I33)*omega1*omega3) / I22
    dw3dt = (T_control[2] - (I22-I11)*omega2*omega1) / I33

    omegaDot = np.array([dw1dt, dw2dt, dw3dt]) #returns the dw/dt full vector
    #qDot = getAmat(omega) @ q

    stateVecDot = np.zeros([12])
    stateVecDot[0:3] = omegaDot
    stateVecDot[3:6] = omega
    #stateVecDot[3:6] = qDot
    #note quaternions used because it creates smooth interpolation for animations. this is called slerp

    ###########################
    # TRAJECTORY
    ###########################
    # unpack variables
    x_LVLH, y_LVLH, z_LVLH, dx_LVLH, dy_LVLH, dz_LVLH = stateVec[6:12]
    
    # Trajectory Control
    dr = np.array([x_LVLH, y_LVLH, z_LVLH])
    dr_ref = interpolate_3d(interp_dx,interp_dy,interp_dz, t)
    dr_error = dr_ref - dr

    u_x, integral_x, prev_error_x, prev_time = pid_control(t, dr_error[0], kPx, kIx, kDx, integral_x, prev_error_x, prev_time_iter)
    u_y, integral_y, prev_error_y, prev_time = pid_control(t, dr_error[1], kPy, kIy, kDy, integral_y, prev_error_y, prev_time_iter)
    u_z, integral_z, prev_error_z, prev_time = pid_control(t, dr_error[2], kPz, kIz, kDz, integral_z, prev_error_z, prev_time_iter)
        
    #print( "input_x: {}".format(u_x))

    if u_x > u_x_thresh: u_x = u_x_max
    elif u_x < -u_x_thresh: u_x = -u_x_max
    else: u_x = 0
    if u_y > u_y_thresh: u_y = u_y_max
    elif u_y < -u_y_thresh: u_y = -u_y_max
    else: u_y = 0
    if u_z > u_z_thresh: u_z = u_z_max
    elif u_z < -u_z_thresh: u_z = -u_z_max
    else: u_z = 0

   # print("time: {}".format(t), "ref point: {}".format(dr_ref),"prev point: {}".format(dr) , "input_x: {}".format(u_x))
    #print( "input_x: {}".format(u_x))
    print(t)
    # Hill Eqns, Chaser
    ddx_LVLH = 2 * n * dy_LVLH + 3 * n**2 * x_LVLH  + u_x
    ddy_LVLH = -2 * n * dx_LVLH + u_y
    ddz_LVLH = -n**2 * z_LVLH + u_z
    a_LVLH = np.array([ddx_LVLH, ddy_LVLH, ddz_LVLH])
    a_LVLH = a_LVLH + f # m/sec^2
 
    stateVecDot[6:12] = dx_LVLH,dy_LVLH,dz_LVLH,a_LVLH[0],a_LVLH[1],a_LVLH[2]
        
    return stateVecDot

###############################################
#                   INPUTS                    #
###############################################
InertMat = np.array([[1,0,0], [0,1,0],[0,0,1]]) #inertial matrix
w0 = np.array([np.deg2rad(0.1),np.deg2rad(0.2),np.deg2rad(0.3)]) #initial angular velocity
theta0 = np.array([30,20,10]) #initial attitude in degrees (roll, pitch, yaw)

def T_ext_func(t): #define the thrust over time in body frame
   T1 = 0
   T2 = 0
   T3 = 0
   return np.array([T1, T2, T3])

t = 60
tspan = np.array([0, t]) #spans one minute (start and stop)
dt = 0.01 #timestep in seconds

triangleInequality(InertMat) #checks that the object exists
theta0 = theta0 * 2*np.pi/360 #convert attitude to radians

#TRAJECTORY INPUTS
R_Earth = 6378
a = 400 + 6378 # semi major axis
I = np.deg2rad(0)
e = 0
f = np.deg2rad(0)
RAAN = np.deg2rad(0)
AOP = np.deg2rad(0)
mu = 398600 # Earth gravitational param
tau = np.sqrt(a**3 * 4 * np.pi**2 / mu) # orbital period
n = 2*np.pi / tau # mean motion

rT_ECI0, vT_ECI0 = sv_from_coe([a, e, RAAN, I, AOP, f], mu) # state vector of target sc, initially

# Chaser ICs
x0 = 1.5
y0 = -0.2
z0 = 0.2
dx0 = 0
dy0 = 0
dz0 = 0

rC_LVLH0 = np.array([x0, y0, z0])
dr1 = np.array([0.5,0,0]) # end position
vC_LVLH0 = cw_calc_dv0(rC_LVLH0,dr1, t, n)
rCrel_ECI0, vCrel_ECI0 = LVLH2ECI(rT_ECI0, vT_ECI0, rC_LVLH0, vC_LVLH0)
rC_ECI0 = rCrel_ECI0/1000 + rT_ECI0 # chaser position in ECI, in km
vC_ECI0 = vCrel_ECI0/1000 + vT_ECI0 # in km/sec

dr0 = rC_LVLH0
dv0 = vC_LVLH0
NumWPs = 1
drvec = np.array([dr1])
tvec = np.array([t])
# This is the planned trajectory
Traj, deltavs = PlanTrajectory(NumWPs, drvec, tvec, dr0, dv0, dt) # Traj has ith row: drx,dry,drz,dvx,dvy,dvz,t

# Define Pyramid geometry
base = [(1.5, 0.4, 0.4), (1.5, -0.4, 0.4), (1.5, -0.4, -0.4), (1.5, 0.4, -0.4)]
apex = (0,0,0)
# Find points outside pyramid
results = check_points_in_pyramid(Traj[:,0:3], base, apex)

# Trajectory Interpolation Functions
interp_dx = interp1d(Traj[:,6], Traj[:,0], kind='cubic', fill_value="extrapolate") # inputs: time and position
interp_dy = interp1d(Traj[:,6], Traj[:, 1], kind='cubic', fill_value="extrapolate")
interp_dz = interp1d(Traj[:,6], Traj[:, 2], kind='cubic', fill_value="extrapolate")

# Trajectory PID Parameters
integral_x = 0
prev_error_x = 0
integral_y = 0
prev_error_y = 0
integral_z = 0
prev_error_z = 0
prev_time = 0

kPx = 0#4
kIx = 0
kDx = 0#1
kPy = 0#4
kIy = 0
kDy = 0#1
kPz = 0#4
kIz = 0
kDz = 0#1

u_x_max = 10e-3 # m/sec^2
u_y_max = 10e-3 # m/sec^2
u_z_max = 10e-3 # m/sec^2

u_x_thresh = 5e-4
u_y_thresh = 5e-4
u_z_thresh = 5e-4

# Attitude PID Parameters [roll pitch yaw]
integral_roll = 0
prev_error_roll = 0
integral_pitch = 0
prev_error_pitch = 0
integral_yaw = 0
prev_error_yaw = 0

kP_roll = 0#1*0.8#2*0.8
kI_roll = 0
kD_roll = 0#0.1*1*270#2*0.1*40
kP_pitch = 0#1*0.8#2*0.8
kI_pitch = 0
kD_pitch = 0#0.1*1*270#1*0.1*40
kP_yaw = 0#1*0.8#2*0.8
kI_yaw = 0
kD_yaw = 0#0.1*1*270#1*0.1*40

u_roll_max = 10e-5 # m/sec^2
u_pitch_max = 10e-5 # rad/sec^2
u_yaw_max = 10e-5 # m/sec^2

u_roll_thresh = 5e-5
u_pitch_thresh = 5e-5
u_yaw_thresh = 5e-5

roll_ref = np.deg2rad(30)
pitch_ref = np.deg2rad(50)
yaw_ref = np.deg2rad(40)
C_ref = eulerToDCM(np.array([roll_ref,pitch_ref,yaw_ref]))
q_ref = DCMtoQuaternion(C_ref)

###############################################
#                 PROCESSING                  #
###############################################
t_eval = np.arange(tspan[0], tspan[1]+dt, dt) #when to store state matrix
q0 = DCMtoQuaternion(eulerToDCM(theta0)) #get intial quaternion

#initialise the initial state vector (made up of angular velos and quaternions at each timestep)
isv = np.zeros([25])
isv[0:3] = w0
isv[3:7] = q0
isv[7:26] = rT_ECI0[0],rT_ECI0[1],rT_ECI0[2],vT_ECI0[0],vT_ECI0[1],vT_ECI0[2], rC_ECI0[0],rC_ECI0[1],rC_ECI0[2],vC_ECI0[0],vC_ECI0[1],vC_ECI0[2], rC_LVLH0[0],rC_LVLH0[1],rC_LVLH0[2],vC_LVLH0[0],vC_LVLH0[1],vC_LVLH0[2]

fullSolution = sc.integrate.solve_ivp(TrajandAtt, tspan, isv, method='RK45', t_eval = t_eval, args=(T_ext_func,interp_dx,interp_dy,interp_dz), rtol=1e-10)

omegaVec = fullSolution.y[0:3, :] #the .y exctracts just the omegas over the tspan
qs = fullSolution.y[3:7, :]
r_ECI_T = np.array([fullSolution.y[7], fullSolution.y[8], fullSolution.y[9]])
v_ECI_T = np.array([fullSolution.y[10], fullSolution.y[11], fullSolution.y[12]])
r_ECI_C = np.array([fullSolution.y[13], fullSolution.y[14], fullSolution.y[15]])
v_ECI_C = np.array([fullSolution.y[16], fullSolution.y[17], fullSolution.y[18]])
r_LVLH_C = np.array([fullSolution.y[19], fullSolution.y[20], fullSolution.y[21]])
v_LVLH_C = np.array([fullSolution.y[22], fullSolution.y[23], fullSolution.y[24]])

t_eval = fullSolution.t

rollVec = np.zeros([len(qs[0,:]),1])
pitchVec = np.zeros([len(qs[0,:]),1])
yawVec = np.zeros([len(qs[0,:]),1])
rollrateVec = np.zeros([len(qs[0,:]),1])
pitchrateVec = np.zeros([len(qs[0,:]),1])
yawrateVec = np.zeros([len(qs[0,:]),1])

# convert to degrees
for ii in range(0,len(qs[0,:])):
    tii = ii*dt
    rollVec[ii] = np.rad2deg(theta0[0]) + np.rad2deg(w0[0])*tii
    pitchVec[ii] = np.rad2deg(theta0[1]) + np.rad2deg(w0[1])*tii
    yawVec[ii] = np.rad2deg(theta0[2]) + np.rad2deg(w0[2])*tii
    

#find the error in the norm of the quaternions from 1
qErr = np.zeros([len(qs[0,:])])
for i in range(len(qs[0,:])):
    qErr[i] = 1 - norm(qs[:,i])


###############################################
#                  PLOTTING                   #
###############################################
diagnosticsPlt = True
matplotlibPlt = False
pybulletPlt = True
acc = 30 #accelerates the time for the dynamic plotting


#centroid = np.array([1,0,0])
print(fullSolution.message)
if diagnosticsPlt:
    # Create a figure with two axes side by side
    fig1, axs = plt.subplots(1, 2, figsize=(15, 5))  # 1 row, 2 columns

    ax1 = axs[0]  # First subplot (Angular velocity)
    ax2 = axs[1]  # Second subplot (Quaternion variation)

    # Plot angular velocities
    ax1.set_title('Angular Velocity Variation')
    ax1.plot(t_eval, np.rad2deg(omegaVec[0]), color='b', label='Roll Rate')
    ax1.plot(t_eval, np.rad2deg(omegaVec[1]), color='r', label='Pitch Rate')
    ax1.plot(t_eval, np.rad2deg(omegaVec[2]), color='g', label='Yaw Rate')
    ax1.grid()
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Angular Velocity (deg/s)')
    ax1.legend()

    # Plot quaternions
    ax2.set_title('Euler Angle Variation')
    ax2.plot(t_eval, rollVec, color='b', label='Roll')
    ax2.plot(t_eval, pitchVec, color='r', label='Pitch')
    ax2.plot(t_eval, yawVec, color='g', label='Yaw')
    ax2.grid()
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Angle, deg')
    ax2.legend()

    plt.subplots_adjust(wspace=0.3)  # Adjust horizontal spacing
    #plt.show()


#### --------------------------------------------------------------------------------------------------------------------------------------------------------------
#  TRAJECTORY Plotting
# #### --------------------------------------------------------------------------------------------------------------------------------------------------------------
# plot in lvlh:
fig = plt.figure(figsize=(8, 4))
ax1 = fig.add_subplot(111, projection='3d')
ax1.scatter(0, 0, 0,s=100, marker='.', c='k', label='origin')
ax1.plot(Traj[:,0], Traj[:,1], Traj[:,2], c='g', label="from LVLH" )
ax1.plot(drvec[:,0],drvec[:,1],drvec[:,2], 'bo', label="Waypoint")
ax1.plot(r_LVLH_C[0,:],r_LVLH_C[1,:],r_LVLH_C[2,:], 'k', label="Feedback Control Path")

ax1.set_xlabel('x (r-bar), m')
ax1.set_ylabel('y (v-bar), m')
ax1.set_zlabel('z (h-bar), m')

ax1.set_title("lvlh frame, {} orbits".format(round(t/tau,2)))
plt.grid(True)
ax1.legend()
plot_pyramid_with_points(base, apex, results,ax1)




if diagnosticsPlt:
    # Create a figure with 6 axes (2 rows, 3 columns)
    fig2, axs = plt.subplots(2, 3, figsize=(15, 10))  

    ax1 = axs[0, 0]  # Angular Velocity
    ax2 = axs[0, 1]  # Euler Angles
    ax3 = axs[0, 2]  # Quaternion Norm Error
    ax4 = axs[1, 0]  # Control Torques
    ax5 = axs[1, 1]  # Angular Momentum
    ax6 = axs[1, 2]  # Energy Variation

    ax1.set_title('Chaser Position, x')
    ax1.plot(t_eval, r_LVLH_C[0], color='b')
    ax1.grid()
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Chaser Position, x (m)')

    ax2.set_title('Chaser Position, y')
    ax2.plot(t_eval, r_LVLH_C[1], color='r')
    ax2.grid()
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Chaser Position, y (m)')

    ax3.set_title('Chaser Position, z')
    ax3.plot(t_eval, r_LVLH_C[2], color='k')
    ax3.grid()
    ax3.set_xlabel('Time (s)')
    ax3.set_ylabel('Chaser Position, z (m)')

    ax4.set_title('Chaser Velocity, x')
    ax4.plot(t_eval, v_LVLH_C[0], color='b')
    ax4.grid()
    ax4.set_xlabel('Time (s)')
    ax4.set_ylabel('Chaser Position, x (m/s)')

    ax5.set_title('Chaser Velocity, y')
    ax5.plot(t_eval, v_LVLH_C[1], color='r')
    ax5.grid()
    ax5.set_xlabel('Time (s)')
    ax5.set_ylabel('Chaser Position, y (m/s)')

    ax6.set_title('Chaser Velocity, z')
    ax6.plot(t_eval, v_LVLH_C[2], color='k')
    ax6.grid()
    ax6.set_xlabel('Time (s)')
    ax6.set_ylabel('Chaser Position, z(m/s)')

    # Adjust spacing between plots
    plt.subplots_adjust(wspace=0.4, hspace=0.4)

    


###############################################
#                  SAVE DATA                  #
###############################################
data_dict = {
    "time": t_eval,  # Time column
    "r_LVLH_C_x": r_LVLH_C[0, :], "r_LVLH_C_y": r_LVLH_C[1, :], "r_LVLH_C_z": r_LVLH_C[2, :],  # Position in LVLH
    #"v_LVLH_C_x": v_LVLH_C[0, :], "v_LVLH_C_y": v_LVLH_C[1, :], "v_LVLH_C_z": v_LVLH_C[2, :],  # Velocity in LVLH
    "roll": rollVec.flatten(), "pitch": pitchVec.flatten(), "yaw": yawVec.flatten(),  # Euler angles
    #"roll_rate": rollrateVec.flatten(), "pitch_rate": pitchrateVec.flatten(), "yaw_rate": yawrateVec.flatten()  # Euler rates
}

# Create Pandas DataFrame
df = pd.DataFrame(data_dict)

# Save to CSV
df.to_csv("TestCase_1-4.csv", index=False)  # No index column in the CSV

print("CSV file saved successfully!")

plt.show()