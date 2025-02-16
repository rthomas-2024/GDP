###############################################
#                   IMPORTS                   #
###############################################

import numpy as np
import pybullet as p
import time



###############################################
#                  FUNCTIONS                  #
###############################################

def findJointsToAnimate():
    #prints the name and type of each joint and its index, not necessarily the index that the joints were created in
    numJoints = p.getNumJoints(gantry)

    for i in range(numJoints):
        jointInfo = p.getJointInfo(gantry, i)
        print("Joint index:", jointInfo[0], "Joint name:", jointInfo[1].decode("utf-8"), "Joint type:", jointInfo[2])
def defineMovement(ts):
    yaxisPos = 1.2*np.sin(ts)
    zaxisPos = 0.5*np.sin(ts)
    gimbalFramePos = 0.2*np.sin(ts)

    return yaxisPos, zaxisPos, gimbalFramePos
def animateSystem(yaxisPos, zaxisPos, gimbalFramePos, tmax, dt):
    for i in range(tmax):

        #move y-axis in x direction
        p.setJointMotorControl2(bodyUniqueId=gantry,
                                jointIndex=yaxisPrismaticJointIndex,
                                controlMode=p.POSITION_CONTROL,
                                targetPosition=yaxisPos[i])

        #move z-axis in y direction
        p.setJointMotorControl2(bodyUniqueId=gantry,
                                jointIndex=zaxisPrismaticJointIndex,
                                controlMode=p.POSITION_CONTROL,
                                targetPosition=zaxisPos[i])

        #move gimbal frame in z direction
        p.setJointMotorControl2(bodyUniqueId=gantry,
                                jointIndex=gimbalFramePrismaticJointIndex,
                                controlMode=p.POSITION_CONTROL,
                                targetPosition=gimbalFramePos[i])

        p.stepSimulation()
        time.sleep(dt)



###############################################
#             VARIABLE DEFINITIONS            #
###############################################

#define time the same way it was done in the dynamic model to make future integration easier
tmax = 10 #seconds
tspan = np.array([0, tmax])
dt = 0.01
t_eval = np.arrange(tspan[0], tspan[1]+dt, dt) #this is the array of times we will use


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

#then get the positions of the links over time
yaxisPos, zaxisPos, gimbalFramePos = defineMovement(t_eval)

animate = True #do we want to animate (control), or move with mouse

if animate:
    animateSystem(yaxisPos, zaxisPos, gimbalFramePos, tmax, dt)
else:
    p.setRealTimeSimulation(1)
    while True:
        pass

