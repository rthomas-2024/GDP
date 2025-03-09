###############################################
#                   IMPORTS                   #
###############################################
import numpy as np
import pybullet as p
import time
import matplotlib.pyplot as plt



###############################################
#                  FUNCTIONS                  #
###############################################
def findJointsToAnimate():
    #prints the name and type of each joint and its index, not necessarily the index that the joints were created in
    numJoints = p.getNumJoints(gantry)

    for i in range(numJoints):
        jointInfo = p.getJointInfo(gantry, i)
        print("Joint index:", jointInfo[0], "Joint name:", jointInfo[1].decode("utf-8"), "Joint type:", jointInfo[2])
def definePosition(t):
    yaxisPos = 1.2*np.sin(t)
    zaxisPos = 0.3*np.sin(t)
    gimbalHolderPos = 0.4*np.sin(t)

    return yaxisPos, zaxisPos, gimbalHolderPos
def animateSystem(tmax):
    #get the simulation start time to ensure the simulation only runs for tmax amount of time
    simulationStart = time.time()

    while True:
        simTime = time.time() - simulationStart

        #end the simulation at tmax
        if simTime >= tmax:
            break

        yaxisPos, zaxisPos, gimbalHolderPos = definePosition(simTime)

        #move y-axis in x direction
        p.setJointMotorControl2(bodyUniqueId=gantry,
                                jointIndex=yaxisPrismaticJointIndex,
                                controlMode=p.POSITION_CONTROL,
                                targetPosition=yaxisPos)

        #move z-axis in y direction
        p.setJointMotorControl2(bodyUniqueId=gantry,
                                jointIndex=zaxisPrismaticJointIndex,
                                controlMode=p.POSITION_CONTROL,
                                targetPosition=zaxisPos)

        #move gimbal frame in z direction
        p.setJointMotorControl2(bodyUniqueId=gantry,
                                jointIndex=gimbalHolderPrismaticJointIndex,
                                controlMode=p.POSITION_CONTROL,
                                targetPosition=gimbalHolderPos)

        p.stepSimulation()
        time.sleep(1/240) #max refresh rate of pybullet, not necessarily same timestep as is used for plotting but of course same function
def getConstrainedPos(yaxisPosFull, zaxisPosFull, gimbalHolderPosFull, numSteps):
    yaxisJointInfo = p.getJointInfo(gantry,yaxisPrismaticJointIndex)
    yaxisJointLowerLimit = yaxisJointInfo[8]
    yaxisJointUpperLimit = yaxisJointInfo[9]

    zaxisJointInfo = p.getJointInfo(gantry,zaxisPrismaticJointIndex)
    zaxisJointLowerLimit = zaxisJointInfo[8]
    zaxisJointUpperLimit = zaxisJointInfo[9]

    gimbalHolderJointInfo = p.getJointInfo(gantry,gimbalHolderPrismaticJointIndex)
    gimbalHolderJointLowerLimit = gimbalHolderJointInfo[8]
    gimbalHolderJointUpperLimit = gimbalHolderJointInfo[9]

    yaxisPosConstrained = np.zeros([numSteps])
    zaxisPosConstrained = np.zeros([numSteps])
    gimbalHolderPosConstrained = np.zeros([numSteps])

    for i in range(numSteps):
        if yaxisPosFull[i] >= yaxisJointUpperLimit:
            yaxisPosConstrained[i] = yaxisJointUpperLimit
        elif yaxisPosFull[i] <= yaxisJointLowerLimit:
            yaxisPosConstrained[i] = yaxisJointLowerLimit
        else:
            yaxisPosConstrained[i] = yaxisPosFull[i]

        if zaxisPosFull[i] >= zaxisJointUpperLimit:
            zaxisPosConstrained[i] = zaxisJointUpperLimit
        elif zaxisPosFull[i] <= zaxisJointLowerLimit:
            zaxisPosConstrained[i] = zaxisJointLowerLimit
        else:
            zaxisPosConstrained[i] = zaxisPosFull[i]

        if gimbalHolderPosFull[i] >= gimbalHolderJointUpperLimit:
            gimbalHolderPosConstrained[i] = gimbalHolderJointUpperLimit
        elif gimbalHolderPosFull[i] <= gimbalHolderJointLowerLimit:
            gimbalHolderPosConstrained[i] = gimbalHolderJointLowerLimit
        else:
            gimbalHolderPosConstrained[i] = gimbalHolderPosFull[i]

    return yaxisPosConstrained, zaxisPosConstrained, gimbalHolderPosConstrained
def getVelos(yaxisPos, zaxisPos, gimbalHolderPos, dt):
    yaxisVelos = np.gradient(yaxisPos, dt)
    zaxisVelos = np.gradient(zaxisPos, dt)
    gimbalHolderVelos = np.gradient(gimbalHolderPos, dt)
    
    return yaxisVelos, zaxisVelos, gimbalHolderVelos
def getAccns(yaxisVelos, zaxisVelos, gimbalHolderVelos, dt):
    yaxisAccns = np.gradient(yaxisVelos, dt)
    zaxisAccns = np.gradient(zaxisVelos, dt)
    gimbalHolderAccns = np.gradient(gimbalHolderVelos, dt)
    
    return yaxisAccns, zaxisAccns, gimbalHolderAccns

###############################################
#             VARIABLE DEFINITIONS            #
###############################################
#define time the same way it was done in the dynamic model to make future integration easier
tmax = 30 #seconds
dt = 1/240 #to get 240 hz
numSteps = int(tmax/dt) + 1
t_eval = np.linspace(0, tmax, numSteps)



###############################################
#             PYBULLET INITIATION             #
###############################################
p.connect(p.GUI)
p.setGravity(0,0,-9.81)
#p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0) #not showing GUI to make animations look nicer

gantry = p.loadURDF("Gantry_CAD.urdf", basePosition=np.array([0,0,0]))

#assign the correct joint number to each joint MAY NEED UPDATING IF MORE JOINTS ADDED
findJointsToAnimate()
yaxisPrismaticJointIndex = 1
zaxisPrismaticJointIndex = 2
gimbalHolderPrismaticJointIndex = 3

###############################################
#                 PROCESSING                  #
###############################################
#initialise plotting position arrays
yaxisPosFull = np.zeros([numSteps])
zaxisPosFull = np.zeros([numSteps])
gimbalHolderPosFull = np.zeros([numSteps])

#fill position plotting arrays
for i in range(numSteps):
    yaxisPosFull[i], zaxisPosFull[i], gimbalHolderPosFull[i] = definePosition(t_eval[i])

#the previous loop uses the full movement function, bit of course the movement is constrained by the joint limits
yaxisPosConstrained, zaxisPosConstrained, gimbalHolderPosConstrained = getConstrainedPos(yaxisPosFull, zaxisPosFull, gimbalHolderPosFull, numSteps)

yaxisVelos, zaxisVelos, gimbalHolderVelos = getVelos(yaxisPosConstrained, zaxisPosConstrained, gimbalHolderPosConstrained, dt)

yaxisAccns, zaxisAccns, gimbalHolderAccns = getAccns(yaxisVelos, zaxisVelos, gimbalHolderVelos, dt)

###############################################
#                  PLOTTING                   #
###############################################
fig1, axs = plt.subplots(2, 2, figsize=(15,10))
ax1 = axs[0,0] #position
ax2 = axs[0,1] #velocity
ax3 = axs[1,0] #acceleration
ax4 = axs[1,1] #spare for now

ax1.set_title("Position")
ax1.plot(t_eval, yaxisPosFull, "r--", label="y-axis (Unconstrained)")
ax1.plot(t_eval, yaxisPosConstrained, "r-", label="y-axis (Constrained)")
ax1.plot(t_eval, zaxisPosFull, "g--")
ax1.plot(t_eval, zaxisPosConstrained, "g-", label="z-axis")
ax1.plot(t_eval, gimbalHolderPosFull, "b--")
ax1.plot(t_eval, gimbalHolderPosConstrained, "b-", label="Gimbal Bracket")
ax1.grid()
ax1.legend()
ax1.set_xlabel("Time (s)")
ax1.set_ylabel("Displacement from origin (m)")
ax1.set_xlim(0,tmax)
ax1.set_ylim(-1.5, 1.5)

ax2.set_title("Velocity")
ax2.plot(t_eval, yaxisVelos, "r-", label="y-axis (x direction)")
ax2.plot(t_eval, zaxisVelos, "g-", label="z-axis (y direction)")
ax2.plot(t_eval, gimbalHolderVelos, "b-", label="Gimbal Bracket (z direction)")
ax2.grid()
ax2.legend()
ax2.set_xlabel("Time (s)")
ax2.set_ylabel("Velocity (m/s)")
ax2.set_xlim(0,tmax)
#ax2.set_ylim(-1.5, 1.5)

ax3.set_title("Acceleration")
ax3.plot(t_eval, yaxisAccns, "r-", label="y-axis (x direction)")
ax3.plot(t_eval, zaxisAccns, "g-", label="z-axis (y direction)")
ax3.plot(t_eval, gimbalHolderAccns, "b-", label="Gimbal Bracket (z direction)")
ax3.grid()
ax3.legend()
ax3.set_xlabel("Time (s)")
ax3.set_ylabel("Acceleration (m/s)")
ax3.set_xlim(0,tmax)
#ax3.set_ylim(-1.5, 1.5)

plt.subplots_adjust(wspace=0.25, hspace=0.3)
plt.show()



###############################################
#                  ANIMATION                  #
###############################################

animate = True #do we want to animate (control), or move with mouse

if animate:
    #call the animate function here
    animateSystem(tmax)
else:
    p.setRealTimeSimulation(1)
    while True:
        pass
