###############################################
#                  IMPORTS                   #
###############################################
import numpy as np
import matplotlib.pyplot as plt
import scipy as sc
from scipy.linalg import norm
import sys



###############################################
#                 FUNCTIONS                   #
###############################################
# UTILITY FUNCTIONS
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

#TRANSFORMATION FUNCTIONS
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

#CORE FUNCTIONS
def EulerEquations(t, stateVec, T_ext_func):
    #I is the full inertial matrix, and omega is an angular velocity vector
    I11 = I[0,0]
    I22 = I[1,1]
    I33 = I[2,2]

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



###############################################
#                   INPUTS                    #
###############################################
I = np.array([[1,0,0], [0,1,0],[0,0,1]]) #inertial matrix
w0 = np.array([0.3,1,0]) #initial angular velocity
theta0 = np.array([30,15,10]) #initial attitude in degrees (roll, pitch, yaw)

def T_ext_func(t): #define the thrust over time in body frame
   T1 = 0
   T2 = 0
   T3 = 0
   return np.array([T1, T2, T3])

tspan = np.array([0, 120]) #spans one minute (start and stop)
dt = 0.01 #timestep in seconds

triangleInequality(I) #checks that the object exists
theta0 = theta0 * 2*np.pi/360 #convert attitude to radians



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
#           FIND ATTITUDE OVER TIME           #
###############################################

#thetas = getAttitudes(theta0, t_eval, betas)

#fig4 = plt.figure(figsize=(10,10))
#plt.plot(t_eval, thetas[:,0]*360/(2*np.pi), color='b', label='theta1')
#plt.plot(t_eval, thetas[:,1]*360/(2*np.pi), color='r', label='theta2')
#plt.plot(t_eval, thetas[:,2]*360/(2*np.pi), color='g', label='theta3')
#plt.grid()
#plt.xlabel('time (s)')
#plt.ylabel('Angle (deg)')
#plt.legend()
#plt.show()



###############################################
#               CUBE PLOTTING                 #
###############################################
fig2 = plt.figure(figsize = (10, 10))
ax = plt.axes(projection = '3d')

centroid = np.array([0.5,0.5,0.5])
length = 1

for i in range(len(t_eval)):
    ax.clear()
    ax.set_xlim(-2, 2)
    ax.set_ylim(-2, 2)
    ax.set_zlim(-2, 2)
    ax.set_xlabel('x')
    ax.set_xlabel('y')
    ax.set_zlabel('z')
    ax.set_title('Cube Plot')
    ax.set_aspect('equal')

    vert = getVertices(centroid, length, qs[:,i])
    plotCube(vert)
    ax.plot3D(centroid[0], centroid[1], centroid[2], marker=".", markersize=10, color="g")

    plt.pause(dt)

plt.show()