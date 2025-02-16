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
    zaxisPos = 0.5*np.sin(t)
    gimbalFramePos = 0.2*np.sin(t)

    return yaxisPos, zaxisPos, gimbalFramePos
def animateSystem(tmax):
    #get the simulation start time to ensure the simulation only runs for tmax amount of time
    simulationStart = time.time()

    while True:
        simTime = time.time() - simulationStart

        #end the simulation at tmax
        if simTime >= tmax:
            break

        yaxisPos, zaxisPos, gimbalFramePos = definePosition(simTime)

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
                                jointIndex=gimbalFramePrismaticJointIndex,
                                controlMode=p.POSITION_CONTROL,
                                targetPosition=gimbalFramePos)

        p.stepSimulation()
        time.sleep(1/240) #max refresh rate of pybullet, not necessarily same timestep as is used for plotting but of course same function



###############################################
#             VARIABLE DEFINITIONS            #
###############################################
#define time the same way it was done in the dynamic model to make future integration easier
tmax = 10 #seconds
dt = 1/240 #to get 240 hz
numSteps = int(tmax/dt) + 1
t_eval = np.linspace(0, tmax, numSteps)



###############################################
#                 PROCESSING                  #
###############################################

#initialise plotting position arrays
yaxisPosPlt = np.zeros([numSteps])
zaxisPosPlt = np.zeros([numSteps])
gimbalFramePosPlt = np.zeros([numSteps])

#fill position plotting arrays
for i in range(numSteps):
    yaxisPosPlt[i], zaxisPosPlt[i], gimbalFramePosPlt[i] = definePosition(t_eval[i])

#the previous loop fills the whole

###############################################
#                  PLOTTING                   #
###############################################
fig1, axs = plt.subplots(2, 2, figsize=(15,10))
ax1 = axs[0,0] #yaxis position
ax2 = axs[0,1] #zaxis position
ax3 = axs[1,0] #gimbal frame position
ax4 = axs[1,1] #spare for now

ax1.set_title("y-axis Position")
ax1.plot(t_eval, yaxisPosPlt, "r--")
ax1.grid()
ax1.set_xlabel("Time (s)")
ax1.set_ylabel("Displacement from origin (m)")
ax1.set_xlim(0,10)
ax1.set_ylim(-1.5, 1.5)

ax2.set_title("z-axis Position")
ax2.plot(t_eval, zaxisPosPlt, "g--")
ax2.grid()
ax2.set_xlabel("Time (s)")
ax2.set_ylabel("Displacement from origin (m)")
ax2.set_xlim(0,10)
ax2.set_ylim(-1.5, 1.5)

ax3.set_title("Gimbal Frame Position")
ax3.plot(t_eval, gimbalFramePosPlt, "b--")
ax3.grid()
ax3.set_xlabel("Time (s)")
ax3.set_ylabel("Displacement from origin (m)")
ax3.set_xlim(0,10)
ax3.set_ylim(-1.5, 1.5)

plt.subplots_adjust(wspace=0.25, hspace=0.3)
plt.show()

###############################################
#                  PYBULLET                   #
###############################################
p.connect(p.GUI)
p.setGravity(0,0,-9.81)
p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0) #not showing GUI to make animations look nicer

gantry = p.loadURDF("Gantry.urdf", basePosition=np.array([0,0,0]))

#assign the correct joint number to each joint
findJointsToAnimate()
yaxisPrismaticJointIndex = 6
zaxisPrismaticJointIndex = 7
gimbalFramePrismaticJointIndex = 8

animate = True #do we want to animate (control), or move with mouse

if animate:
    #call the animate function here
    animateSystem(tmax)
else:
    p.setRealTimeSimulation(1)
    while True:
        pass


