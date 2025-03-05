###############################################
#                  IMPORTS                   #
###############################################
import numpy as np
from numpy import sin as s
from numpy import cos as c
from numpy import pi as pi
import matplotlib.pyplot as plt
import scipy as sc
from scipy.linalg import norm
import sys
import time
import pybullet as p



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

#TRANSFORMATION FUNCTIONS
def quaternionToDCM(beta):
    #beta passed in as a quaternion (4 value vector)
    beta /= np.linalg.norm(beta)
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
            print(sinPitch)
            print(cosPitch)
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

#CORE FUNCTIONS
def EulerEquations(t, stateVec, T_ext_func):
    #I is the full inertial matrix, and omega is an angular velocity vector
    omega = stateVec[0:3]
    q = stateVec[3:7]

    # Control stuff
    global prev_time, integral_x,integral_y,integral_z,prev_error_x,prev_error_y,prev_error_z, integral_roll,integral_pitch,integral_yaw,prev_error_roll,prev_error_pitch,prev_error_yaw
    prev_time_iter = prev_time
    C = quaternionToDCM(q)
    roll, pitch, yaw = DCMtoEuler(C)
    roll_err = roll_ref - roll
    pitch_err = pitch_ref - pitch
    yaw_err = yaw_ref - yaw
    
    u_roll, integral_roll, prev_error_roll, prev_time = pid_control(t, roll_err, kP_roll, kI_roll, kD_roll, integral_roll, prev_error_roll, prev_time_iter)
    u_pitch, integral_pitch, prev_error_pitch, prev_time = pid_control(t, pitch_err, kP_pitch, kI_pitch, kD_pitch, integral_pitch, prev_error_pitch, prev_time_iter)
    u_yaw, integral_yaw, prev_error_yaw, prev_time = pid_control(t, yaw_err, kP_yaw, kI_yaw, kD_yaw, integral_yaw, prev_error_yaw, prev_time_iter)
    print("Pitch: {}".format(pitch))
    print("DCM : {}".format(C))
    print("u_roll: {}".format(u_roll),"u_pitch: {}".format(u_pitch),"u_yaw: {}".format(u_yaw))

    if u_roll > u_roll_thresh: u_roll = u_roll_thresh
    elif u_roll < -u_roll_thresh: u_roll = -u_roll_thresh
    else: u_roll = 0
    if u_pitch > u_pitch_thresh: u_pitch = u_pitch_thresh
    elif u_pitch < -u_pitch_thresh: u_pitch = -u_pitch_thresh
    else: u_pitch = 0
    if u_yaw > u_yaw_thresh: u_yaw = u_yaw_thresh
    elif u_yaw < -u_yaw_thresh: u_yaw = -u_yaw_thresh
    else: u_yaw = 0

    
    omega1, omega2, omega3 = omega
    T1, T2, T3 = T_ext_func(t)
    T1 = T1 + u_roll
    T2 = T2 + u_pitch
    T3 = T3 + u_yaw
    
    T_vec = np.array([T1, T2, T3])
    omega_vec = np.array([omega1, omega2, omega3])

    I_mult_omegaDot = T_vec-np.cross(omega_vec, np.dot(I, omega_vec))

    omegaDot = np.dot(np.linalg.inv(I), I_mult_omegaDot) #returns the dw/dt full vector

    qDot = 0.5 * getAmat(omega) @ q

    #normalise incoming quaternion to minimise quaternion drift while changing each quaternion minimally as to still represent the correct rotation
    q /= np.linalg.norm(q)

    stateVecDot = np.zeros([7])
    stateVecDot[0:3] = omegaDot
    stateVecDot[3:7] = qDot

    #note quaternions used because it creates smooth interpolation for animations. this is called slerp
    print(t)
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

    #going from inertial to body
    for i in range(8):
        verti = vecToPureQuaternion(vertArr[i,:])
        verti = quaternionMult(q, quaternionMult(verti, inverseQuaternion(q)))
        verti = pureQuaternionToVec(verti)
        vertArr[i,:] = verti

    #'move' centroid back to correct position
    vertArr += centroid

    return vertArr
def plottingFunc(vertices, centroidVec, omegaVec, Hvec, plotComponents):
    #plots 'front and back' faces
    #plotComponents is a 4-length vector containing info on whether to plot cube, centroid, angular velocity and angular momentum vectors

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

    if plotComponents[1]: #centroid plotting
        ax.scatter(centroidVec[0], centroidVec[1], centroidVec[2], color="g", s=50, label='Centroid')

    if plotComponents[2]:  # Angular velocity vector plotting
        #print(omegaVec)
        ax.quiver(centroidVec[0], centroidVec[1], centroidVec[2], omegaVec[0], omegaVec[1], omegaVec[2], label = 'Angular Velocity', color='m')

    if plotComponents[3]:  # Angular momentum vector plotting
        #print(Hvec)
        ax.quiver(centroidVec[0], centroidVec[1], centroidVec[2], Hvec[0], Hvec[1], Hvec[2], label = 'Angular Momentum', color='k')
def getAttitudes(ts, qs):
    thetas = np.zeros([len(ts), 3])

    for i in range(len(ts)):
        thetas[i,:] = DCMtoEuler(quaternionToDCM(qs[:,i]))

    return thetas
def pybulletPlot(centroidVec, q, dt):
    p.resetBasePositionAndOrientation(chaser, centroidVec, quatToVisQuat(q))
    time.sleep(dt)
def transformationTesting():
    dcm = np.array([[1,0,0],
                    [0,1,0],
                    [0,0,1]])

    y = 0
    p = 0
    r = 0

    w = 1
    x = 0
    y = 0
    z = 0

    quaternion = np.array([w,x,y,z])

    euler = np.array([np.deg2rad(r),np.deg2rad(p),np.deg2rad(y)])

    print(quaternionToDCM(quaternion))

    print(" ")

    print(DCMtoQuaternion(dcm))
    
    print(" ")

    print(np.flip(np.rad2deg(DCMtoEuler(dcm))))
    
    print(" ")

    print(eulerToDCM(euler))

def pid_control(t, error, Kp, Ki, Kd, integral,previous_error, previous_time):
    
    # Calculate time step dynamically
    dt = t - previous_time
    
    integral += error * dt
    derivative = (error - previous_error) / dt if dt > 0 else 0
    previous_error = error
    previous_time = t
    
    return Kp * error + Ki * integral + Kd * derivative, integral, previous_error, previous_time


###############################################
#                   INPUTS                    #
###############################################
I = np.array([[5,0,0],
              [0,5,0],
              [0,0,5]]) #inertial matrix

w0 = np.array([np.deg2rad(0),np.deg2rad(0),np.deg2rad(0)]) #initial angular velocity
theta0 = np.array([30, 30, 30]) #initial attitude in degrees (roll, pitch, yaw)

def T_ext_func(t): #define the thrust over time in body frame
   T1 = 0
   T2 = 0
   T3 = 0
   return np.array([T1, T2, T3])

tspan = np.array([0, 2000]) #spans one minute (start and stop)
dt = 0.01 #timestep in seconds

triangleInequality(I) #checks that the object exists
theta0 = np.deg2rad(theta0) #convert attitude to radians


###############################################
#                   Control                   #
###############################################

# Attitude PID Parameters [roll pitch yaw]
prev_time = 0
integral_roll = 0
prev_error_roll = 0
integral_pitch = 0
prev_error_pitch = 0
integral_yaw = 0
prev_error_yaw = 0

Ku = 50
Tu = 41
kP_roll = 0.8*Ku
kI_roll = 0
kD_roll = 0.1*Ku*Tu#0.1*0.1*43
kP_pitch = 0.8*Ku
kI_pitch = 0
kD_pitch = 0.1*Ku*Tu
kP_yaw = 0.8*Ku
kI_yaw = 0
kD_yaw = 0.1*Ku*Tu

u_roll_thresh = 5e-2
u_pitch_thresh = 5e-2
u_yaw_thresh = 5e-2

roll_ref = np.deg2rad(0)
pitch_ref = np.deg2rad(60)
yaw_ref = np.deg2rad(40)
C_ref = eulerToDCM(np.array([roll_ref,pitch_ref,yaw_ref]))
q_ref = DCMtoQuaternion(C_ref)


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

fullSolution = sc.integrate.solve_ivp(EulerEquations, tspan, isv, t_eval = t_eval, args=(T_ext_func,), rtol=1e-10)

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


###############################################
#           FIND ATTITUDE OVER TIME           #
###############################################

thetas = getAttitudes(t_eval, qs)
#attitudes are found in radians from the functions, so are converted to dgrees when plotted below

fig4 = plt.figure(figsize=(10,10))
plt.plot(t_eval, np.rad2deg(thetas[:,0]), color='b', label='Roll')
plt.plot(t_eval, np.rad2deg(thetas[:,1]), color='r', label='Pitch')
plt.plot(t_eval, np.rad2deg(thetas[:,2]), color='g', label='Yaw')
plt.grid()
plt.xlabel('time (s)')
plt.ylabel('Angle (deg)')
plt.legend()
#plt.show()



###############################################
#                  PLOTTING                   #
###############################################
diagnosticsPlt = True
matplotlibPlt = True
pybulletPlt = False

cubePlt = True
centroidPlt = True
omegaVecPlt = True
HvecPlt = True

centroid = np.array([0,0,0])


if diagnosticsPlt:
    #PLOT STATES (DIAGNOSTICS)
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

if matplotlibPlt:
    fig2 = plt.figure(figsize = (10, 10))
    ax = plt.axes(projection = '3d')

    length = 1

    for i in range(len(t_eval)):
        ax.clear()
        ax.set_xlim(-2, 2)
        ax.set_ylim(-2, 2)
        ax.set_zlim(-2, 2)
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')
        ax.set_title('Cube Plot')
        ax.set_aspect('equal')

        vert = getVertices(centroid, length, qs[:,i])

        w = omegaVec[:,i]
        H = I @ w

        w_plt = w/norm(w) * length
        H_plt = H/norm(H) * length

        plotComponents = np.array([cubePlt, centroidPlt, omegaVecPlt, HvecPlt])
        plottingFunc(vert, centroid, w_plt, H_plt, plotComponents)

        ax.legend()
        plt.pause(dt)

    plt.show()


###############################################
#            PYBULLET VISUALISATION           #
###############################################

if pybulletPlt:
    p.connect(p.GUI)
    p.setGravity(0,0,0)

    chaser = p.loadURDF("Chaser.urdf", basePosition=np.array([0,0,0]))

    for i in range(len(t_eval)):
        pybulletPlot(centroid, qs[:,i], dt)



###############################################
#            TESTING AND VALIDATION           #
###############################################

#transformationTesting()


###############################################
#             VIRTUAL TWIN TESTING            #
###############################################

# import pybullet as p
# import pybullet_data
# import time
# import numpy as np

# # Initialize PyBullet simulation
# p.connect(p.GUI)
# p.setAdditionalSearchPath(pybullet_data.getDataPath())
# p.setGravity(0, 0, -9.81)

# # Load the URDF file
# gantry_id = p.loadURDF("gantry.urdf", [0, 0, 0], useFixedBase=True)

# # Get the rail joint index
# num_joints = p.getNumJoints(gantry_id)
# rail_joint_index = None
# for i in range(num_joints):
#     joint_info = p.getJointInfo(gantry_id, i)
#     joint_name = joint_info[1].decode("utf-8")
#     print(f"Joint {i}: {joint_name}")  # Debugging: Print all joint names
#     if "rail_slide" in joint_name:
#         rail_joint_index = i
#         break

# # Ensure the rail joint is found and properly controlled
# if rail_joint_index is not None:
#     print(f"Rail joint found at index {rail_joint_index}")
    
#     # Reset joint state for clean movement
#     p.resetJointState(gantry_id, rail_joint_index, targetValue=0)
    
#     # Enable force-based control instead of velocity control
#     p.setJointMotorControl2(
#         bodyUniqueId=gantry_id,
#         jointIndex=rail_joint_index,
#         controlMode=p.TORQUE_CONTROL,
#         force=0
#     )
    
#     for step in range(1000):
#         target_force = 50.0 * np.sin(step * 0.02)  # Oscillate rail movement using force
#         print(f"Step {step}: Applying force {target_force:.3f}")
        
#         p.setJointMotorControl2(
#             bodyUniqueId=gantry_id,
#             jointIndex=rail_joint_index,
#             controlMode=p.TORQUE_CONTROL,
#             force=target_force
#         )
#         p.stepSimulation()
#         time.sleep(1./240.)
# else:
#     print("Error: Rail joint not found in URDF.")

# p.disconnect()

