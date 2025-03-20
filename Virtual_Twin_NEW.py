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
def getMotorFrequencies(motorControl_t):
    motorDirection = np.zeros(6)
    motorFrequency = np.zeros(6)

    for i in range(6):
        k = 3*i #start of the relevant part of the motor control array for each motor
        if i==2:
            if motorControl_t[k] == 0:
                #we get a div by 0 error if the OCR is 0. of course if OCR 0 then frequency should be 0
                motorFrequency[i] = 0
            else:
                motorFrequency[i] = Fclock[i]/(2*motorControl_t[k]*PRE2[motorControl_t[k+2]])

            motorDirection[i] = motorControl_t[k+1]

        else:
            if motorControl_t[k] == 0:
                motorFrequency[i] = 0
            else:
                motorFrequency[i] = Fclock[i]/(2*motorControl_t[k]*PREnormal[motorControl_t[k+2]])

            motorDirection[i] = motorControl_t[k+1]

    return motorFrequency, motorDirection
def getSpeeds(t):
    step_dm = int(t//dt_dm) #the time-step of the dynamic model
    motorControl_t = motorControlArray[step_dm,:]
    motorFreqs, motorDirns = getMotorFrequencies(motorControl_t)

    #x (800 steps per rotation)
    xSpeed = motorDirns[0]*motorFreqs[0]*(40e-3/800) #m/s

    #y (400 steps per rotation)
    ySpeed = motorDirns[1]*motorFreqs[1]*(3e-3/400) #m/s

    #z (400 steps per rotation)
    zSpeed = motorDirns[2]*motorFreqs[2]*(4e-3/400) #m/s

    #roll (400 steps per rotation)
    rollSpeed = motorDirns[3]*motorFreqs[3]*(2*np.pi*(1/15)/400) #rad/s

    #pitch (400 steps per rotation)
    pitchSpeed = motorDirns[4]*motorFreqs[4]*(2*np.pi*(3/110)/400) #rad/s

    #yaw (400 steps per rotation)
    yawSpeed = motorDirns[5]*motorFreqs[5]*(2*np.pi*0.02/400) #rad/s

    return xSpeed, ySpeed, zSpeed, rollSpeed, pitchSpeed, yawSpeed
def definePosition(t, tDiff):
    global yaxisPos_prev, zaxisPos_prev, gimbalHolderPos_prev, pitchAxisPos_prev, rollAxisPos_prev, endEffInterfacePos_prev

    manualSpeeds = True

    if manualSpeeds:
        yaxisSpeed = 0
        zaxisSpeed = 0
        gimbalHolderSpeed = 0
        pitchAxisSpeed = 0
        rollAxisSpeed = 0
        endEffInterfaceSpeed = 2
    else:
        yaxisSpeed, zaxisSpeed, gimbalHolderSpeed, pitchAxisSpeed, rollAxisSpeed, endEffInterfaceSpeed = getSpeeds(t)

    yaxisPos = yaxisPos_prev + tDiff*yaxisSpeed
    zaxisPos = zaxisPos_prev + tDiff*zaxisSpeed
    gimbalHolderPos = gimbalHolderPos_prev + tDiff*gimbalHolderSpeed
    pitchAxisPos = pitchAxisPos_prev + tDiff*pitchAxisSpeed
    rollAxisPos = rollAxisPos_prev + tDiff*rollAxisSpeed
    endEffInterfacePos = endEffInterfacePos_prev + tDiff*endEffInterfaceSpeed

    #wrap end eff angle as it is 'unlimited' and should be kept within +/- pi
    endEffInterfacePos = (endEffInterfacePos + np.pi) % (2*np.pi) - np.pi

    yaxisPos_prev, zaxisPos_prev, gimbalHolderPos_prev, pitchAxisPos_prev, rollAxisPos_prev, endEffInterfacePos_prev = (yaxisPos, zaxisPos, gimbalHolderPos, pitchAxisPos, rollAxisPos, endEffInterfacePos)

    return yaxisPos, zaxisPos, gimbalHolderPos, pitchAxisPos, rollAxisPos, endEffInterfacePos    
def plot_joint_positions(ts, yaxisPosPlt, zaxisPosPlt, gimbalHolderPosPlt, pitchAxisPosPlt, rollAxisPosPlt, endEffectorPosPlt):
    fig, axes = plt.subplots(6, 1, figsize=(8, 12), sharex=True)  # 6 rows, 1 column, shared x-axis

    # Titles for each subplot
    titles = [
        "Y-Axis Position",
        "Z-Axis Position",
        "Gimbal Holder Position",
        "Pitch Axis Position",
        "Roll Axis Position",
        "End Effector Interface Position"]

    #Display rotations in degrees not radians
    pitchAxisPosPlt = np.rad2deg(pitchAxisPosPlt)
    rollAxisPosPlt = np.rad2deg(rollAxisPosPlt)
    endEffectorPosPlt = np.rad2deg(endEffectorPosPlt)

    # Data for each subplot
    positions = [yaxisPosPlt, zaxisPosPlt, gimbalHolderPosPlt, pitchAxisPosPlt, rollAxisPosPlt, endEffectorPosPlt]

    # Plot each joint position on a separate axis
    for i, ax in enumerate(axes):
        ax.plot(ts, positions[i], label=titles[i], color='b')
        ax.set_ylabel("Position")
        ax.set_title(titles[i])
        ax.legend()
        ax.grid(True)

    axes[-1].set_xlabel("Time (s)")  # X-axis label only on the last plot

    plt.tight_layout()  # Adjust spacing
    plt.show()
def animateSystem(tmax):
    #initialise lists for plotting
    ts = []
    yaxisPosPlt = []
    zaxisPosPlt = []
    gimbalHolderPosPlt = []
    pitchAxisPosPlt = []
    rollAxisPosPlt = []
    endEffectorPosPlt = []

    #get upper and lower bounds for each joint
    yaxisJointInfo = p.getJointInfo(gantry,yaxisPrismaticJointIndex)
    yaxisJointLowerLimit = yaxisJointInfo[8]
    yaxisJointUpperLimit = yaxisJointInfo[9]

    zaxisJointInfo = p.getJointInfo(gantry,zaxisPrismaticJointIndex)
    zaxisJointLowerLimit = zaxisJointInfo[8]
    zaxisJointUpperLimit = zaxisJointInfo[9]

    gimbalHolderJointInfo = p.getJointInfo(gantry,gimbalHolderPrismaticJointIndex)
    gimbalHolderJointLowerLimit = gimbalHolderJointInfo[8]
    gimbalHolderJointUpperLimit = gimbalHolderJointInfo[9]

    pitchAxisJointInfo = p.getJointInfo(gantry,pitchAxisRevoluteJointIndex)
    pitchAxisJointLowerLimit = pitchAxisJointInfo[8]
    pitchAxisJointUpperLimit = pitchAxisJointInfo[9]

    rollAxisJointInfo = p.getJointInfo(gantry,rollAxisRevoluteJointIndex)
    rollAxisJointLowerLimit = rollAxisJointInfo[8]
    rollAxisJointUpperLimit = rollAxisJointInfo[9]
    #shouldnt need this for end effector as no limit switches placed


    #initialise limit switch activation checker
    #0 corresponds to no activation, 1-5 correspond to x, y, z, yaw, pitch (no limit switch on roll)
    limitSwitch = 0
    limitSwitchTime = 0
    limitSwitchArr = np.array(["", "x-axis", "y-axis", "z-axis", "yaw", "pitch"])
    limitUpperLower = ""
    limitBreakMode = False #when set to true, hitting a limit switch will end the simulation
    switchHit = False #to ensure that only the first limit switch hit (most important one) is recorded and the correct time is stored

    #get the simulation start time to ensure the simulation only runs for tmax amount of time
    simulationStart = time.time()
    t_prev = 0

    #begin the animation loop
    while True:
        simTime = time.time() - simulationStart

        #end the simulation at tmax or if a limit switch hit if in that mode
        if simTime >= tmax or (limitSwitch != 0 and limitBreakMode):
            #even if limit break mode off, will display if a limit switch hit
            if limitSwitch != 0:
                print(limitUpperLower + " " + limitSwitchArr[limitSwitch] + " limit switch hit at " + str(round(limitSwitchTime,2)) + "s")
            
            plot_joint_positions(ts, yaxisPosPlt, zaxisPosPlt, gimbalHolderPosPlt, pitchAxisPosPlt, rollAxisPosPlt, endEffectorPosPlt)
            break

        #get new positions of each link
        yaxisPos, zaxisPos, gimbalHolderPos, pitchAxisPos, rollAxisPos, endEffInterfacePos = definePosition(simTime, simTime-t_prev)
        
        #get relevant states of each link (subject to change to world coordinate system soon)
        yaxisCurrentPos = p.getJointState(gantry, yaxisPrismaticJointIndex)[0]
        zaxisCurrentPos = p.getJointState(gantry, zaxisPrismaticJointIndex)[0]
        gimbalHolderCurrentPos = p.getJointState(gantry, gimbalHolderPrismaticJointIndex)[0]
        pitchAxisCurrentPos = p.getJointState(gantry, pitchAxisRevoluteJointIndex)[0]
        rollAxisCurrentPos = p.getJointState(gantry, rollAxisRevoluteJointIndex)[0]
        endEffInterfaceCurrentPos = p.getJointState(gantry, endEffInterfaceRevoluteJointIndex)[0]

        #append the current positions to the lists for plotting
        ts.append(simTime)
        yaxisPosPlt.append(yaxisCurrentPos)
        zaxisPosPlt.append(zaxisCurrentPos)
        gimbalHolderPosPlt.append(gimbalHolderCurrentPos)
        pitchAxisPosPlt.append(pitchAxisCurrentPos)
        rollAxisPosPlt.append(rollAxisCurrentPos)
        endEffectorPosPlt.append(endEffInterfaceCurrentPos)

        #move y-axis in x direction
        if yaxisPos >= yaxisJointUpperLimit:
            #if joint is outside of limits, position will of course not update
            limitSwitch = 1
            limitSwitchTime = simTime
            limitUpperLower = "Upper"
        elif yaxisPos <= yaxisJointLowerLimit:
            limitSwitch = 1
            limitSwitchTime = simTime
            limitUpperLower = "Lower"
        else:
            p.resetJointState(gantry, yaxisPrismaticJointIndex, yaxisPos)


        #KEEPING THIS CODE AS CHAT SAID IT COULD BE USEFUL FOR AVOIDING DRIFT, DOESNT SEEM TO AFFECT IT BUT KEEPING ALL THE SAME
        # p.setJointMotorControl2(bodyUniqueId=gantry,
        #                         jointIndex=yaxisPrismaticJointIndex,
        #                         controlMode=p.POSITION_CONTROL,
        #                         targetPosition=yaxisPos)


        #move z-axis in y direction
        if zaxisPos >= zaxisJointUpperLimit:
            #if joint is outside of limits, position will of course not update
            limitSwitch = 2
            limitSwitchTime = simTime
            limitUpperLower = "Upper"
        elif zaxisPos <= zaxisJointLowerLimit:
            limitSwitch = 2
            limitSwitchTime = simTime
            limitUpperLower = "Lower"
        else:
            p.resetJointState(gantry, zaxisPrismaticJointIndex, zaxisPos)

        #move gimbal in z direction
        if gimbalHolderPos >= gimbalHolderJointUpperLimit:
            #if joint is outside of limits, position will of course not update
            limitSwitch = 3
            limitSwitchTime = simTime
            limitUpperLower = "Upper"
        elif gimbalHolderPos <= gimbalHolderJointLowerLimit:
            limitSwitch = 3
            limitSwitchTime = simTime
            limitUpperLower = "Lower"
        else:
            p.resetJointState(gantry, gimbalHolderPrismaticJointIndex, gimbalHolderPos)

        #rotate the pitch axis in yaw
        if pitchAxisPos >= pitchAxisJointUpperLimit:
            #if joint is outside of limits, position will of course not update
            limitSwitch = 4
            limitSwitchTime = simTime
            limitUpperLower = "Upper"
        elif pitchAxisPos <= pitchAxisJointLowerLimit:
            limitSwitch = 4
            limitSwitchTime = simTime
            limitUpperLower = "Lower"
        else:
            p.resetJointState(gantry, pitchAxisRevoluteJointIndex, pitchAxisPos)

        #rotate the roll axis in pitch
        if rollAxisPos >= rollAxisJointUpperLimit:
            #if joint is outside of limits, position will of course not update
            limitSwitch = 5
            limitSwitchTime = simTime
            limitUpperLower = "Upper"
        elif rollAxisPos <= rollAxisJointLowerLimit:
            limitSwitch = 5
            limitSwitchTime = simTime
            limitUpperLower = "Lower"
        else:
            p.resetJointState(gantry, rollAxisRevoluteJointIndex, rollAxisPos)

        #rotate the end effector interface in roll
        p.resetJointState(gantry, endEffInterfaceRevoluteJointIndex, endEffInterfacePos)

        t_prev = simTime
        p.stepSimulation()
        #max refresh rate of pybullet, not necessarily same timestep as is used for plotting but of course same function
        time.sleep(1/240)

# OLD FUNCTIONS, MAY NOT BE USED AT ALLL IN THIS CODE
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
tmax = 1 #seconds
dt_dm = 0.1 #dynamic model dt (dt over which the velocity is changed)

#define paramters used to calculate motor frequency
PREnormal = np.array([1,8,64,256,1024])
PRE2 = np.array([1,8,32,64,128,256,1024])
Fclock = np.array([8e6,16e6,8e6,16e6,16e6,16e6])

#define the intitial position of each link
yaxisPos_prev = 0.3
zaxisPos_prev = 0
gimbalHolderPos_prev = 0
pitchAxisPos_prev = 0
rollAxisPos_prev = 0
endEffInterfacePos_prev = 0

#store the motor controls in an array
motorControlArray = np.array([[33112, 1, 0, 8107, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0],
                              [52000, 1, 0, 5050, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0],
                              [40000, 1, 0, 7000, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0],
                              [45000, 1, 0, 6000, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0]])



###############################################
#             PYBULLET INITIATION             #
###############################################
p.connect(p.GUI)
p.setGravity(0,0,-9.81)
p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0) #not showing GUI to make animations look nicer

gantry = p.loadURDF("Gantry_CAD.urdf", basePosition=np.array([0,0,0]))

#assign the correct joint number to each joint MAY NEED UPDATING IF MORE JOINTS ADDED
findJointsToAnimate()
yaxisPrismaticJointIndex = 1
zaxisPrismaticJointIndex = 2
gimbalHolderPrismaticJointIndex = 3
pitchAxisRevoluteJointIndex = 5
rollAxisRevoluteJointIndex = 6
endEffInterfaceRevoluteJointIndex = 7



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
        # for joint_index in range(8):
        #     link_state = p.getLinkState(gantry, joint_index)
        #     world_position = link_state[0]

        #     # Draw a small red sphere at the joint origin
        #     p.addUserDebugText("O", world_position, textSize=1, textColorRGB=[1, 0, 0])
        #     p.addUserDebugLine(world_position, [world_position[0], world_position[1], world_position[2] + 0.1], [1, 0, 0], 2)
        pass