import numpy as np
import pybullet as p
import time

def findJointsToAnimate():
    #prints the name and type of each joint and its index, not necessarily the index that the joints were created in
    numJoints = p.getNumJoints(gantry)

    for i in range(numJoints):
        jointInfo = p.getJointInfo(gantry, i)
        print("Joint index:", jointInfo[0], "Joint name:", jointInfo[1].decode("utf-8"), "Joint type:", jointInfo[2])
def animateSystem():
    while True:
        t = time.time()
        targetPosn = 1.2*np.sin(t)

        #move y-axis in x direction
        p.setJointMotorControl2(bodyUniqueId=gantry,
                                jointIndex=yaxisPrismaticJointIndex,
                                controlMode=p.POSITION_CONTROL,
                                targetPosition=targetPosn)

        #move z-axis in y direction
        p.setJointMotorControl2(bodyUniqueId=gantry,
                                jointIndex=zaxisPrismaticJointIndex,
                                controlMode=p.POSITION_CONTROL,
                                targetPosition=targetPosn)

        #move gimbal frame in z direction
        p.setJointMotorControl2(bodyUniqueId=gantry,
                                jointIndex=gimbalFramePrismaticJointIndex,
                                controlMode=p.POSITION_CONTROL,
                                targetPosition=targetPosn)


        p.stepSimulation()
        time.sleep(1/240)

p.connect(p.GUI)
p.setGravity(0,0,-9.81)
p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)

gantry = p.loadURDF("Gantry.urdf", basePosition=np.array([0,0,0]))

findJointsToAnimate()
yaxisPrismaticJointIndex = 6
zaxisPrismaticJointIndex = 7
gimbalFramePrismaticJointIndex = 8

animate = True

if animate:
    animateSystem()
else:
    p.setRealTimeSimulation(1)
    while True:
        pass

