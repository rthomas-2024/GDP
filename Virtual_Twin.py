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
def animateSystem(dt, tmax):
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
        time.sleep(dt)



###############################################
#             VARIABLE DEFINITIONS            #
###############################################
#define time the same way it was done in the dynamic model to make future integration easier
tmax = 10 #seconds
dt = 1/240 #to get 240 hz
numSteps = int(tmax/dt) + 1
t_eval = np.linspace(0, tmax, numSteps)


###############################################
#                PYBULLET SETUP               #
###############################################
p.connect(p.GUI)
p.setGravity(0,0,-9.81)
p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0) #not showing GUI to make animations look nicer

gantry = p.loadURDF("Gantry.urdf", basePosition=np.array([0,0,0]))


###############################################
#                 PROCESSING                  #
###############################################
#first assign the correct joint number to each joint
findJointsToAnimate()
yaxisPrismaticJointIndex = 6
zaxisPrismaticJointIndex = 7
gimbalFramePrismaticJointIndex = 8

animate = True #do we want to animate (control), or move with mouse

if animate:
    #call the animate function here
    animateSystem(dt, tmax)
else:
    p.setRealTimeSimulation(1)
    while True:
        pass


###############################################
#                  PLOTTING                   #
###############################################

