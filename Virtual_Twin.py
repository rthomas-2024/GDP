###############################################
#                   IMPORTS                   #
###############################################
import numpy as np
import pybullet as p
import time
import json
import math
import pandas as pd


###############################################
#                  FUNCTIONS                  #
###############################################
def pybulletInit():
    p.connect(p.GUI)
    p.setGravity(0,0,0)
    #p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0) #not showing GUI to make animations look nicer

    gantry = p.loadURDF("Gantry_CAD.urdf", basePosition=np.array([0,0,0]))

    return gantry
def findJointsToAnimate():
    #prints the name and type of each joint and its index, not necessarily the index that the joints were created in
    numJoints = p.getNumJoints(gantry)

    for i in range(numJoints):
        jointInfo = p.getJointInfo(gantry, i)
        print("Joint index:", jointInfo[0], "Joint name:", jointInfo[1].decode("utf-8"), "Joint type:", jointInfo[2])
def getJointLims():
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

    return yaxisJointLowerLimit, yaxisJointUpperLimit, zaxisJointLowerLimit, zaxisJointUpperLimit, gimbalHolderJointLowerLimit, gimbalHolderJointUpperLimit, pitchAxisJointLowerLimit, pitchAxisJointUpperLimit, rollAxisJointLowerLimit, rollAxisJointUpperLimit
def vtToQualisys(vtStateVec):
    x, y, z, r, p, yw = vtStateVec
    
    #offset FROM virtual twin TO qualisys
    x += 0
    y += 0
    z += 0
    r += 0
    p += 0
    yw += 0

    return x, y, z, r, p, yw
def setInitialPosition():
    p.resetJointState(gantry, yaxisPrismaticJointIndex, yaxisPos_prev)
    p.resetJointState(gantry, zaxisPrismaticJointIndex, zaxisPos_prev)
    p.resetJointState(gantry, gimbalHolderPrismaticJointIndex, gimbalHolderPos_prev)
    p.resetJointState(gantry, pitchAxisRevoluteJointIndex, pitchAxisPos_prev)
    p.resetJointState(gantry, rollAxisRevoluteJointIndex, rollAxisPos_prev)
    p.resetJointState(gantry, endEffInterfaceRevoluteJointIndex, endEffInterfacePos_prev)

    return
def getMotorFrequencies(motorControl_t):
    motorDirection = np.zeros(6)
    motorFrequency = np.zeros(6)

    for i in range(6):
        k = 3*i #start of the relevant part of the motor control array for each motor
        if i==4: #4 corresponds to pitch, controlled by motor 2
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

        #0 represents the negative direction
        for i in range(6):
            if motorDirection[i] == 0:
                motorDirection[i] = -1

    return motorFrequency, motorDirection
def getSpeeds(step_dm):
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

    #need to get previous speeds to ensure no drift occurs by overshooting the 0.1s timestep
    try:
        motorControl_tPrev = motorControlArray[step_dm - 1,:]
        motorFreqs, motorDirns = getMotorFrequencies(motorControl_tPrev)

        #x (800 steps per rotation)
        xSpeedPrev = motorDirns[0]*motorFreqs[0]*(40e-3/800) #m/s

        #y (400 steps per rotation)
        ySpeedPrev = motorDirns[1]*motorFreqs[1]*(3e-3/400) #m/s

        #z (400 steps per rotation)
        zSpeedPrev = motorDirns[2]*motorFreqs[2]*(4e-3/400) #m/s

        #roll (400 steps per rotation)
        rollSpeedPrev = motorDirns[3]*motorFreqs[3]*(2*np.pi*(1/15)/400) #rad/s

        #pitch (400 steps per rotation)
        pitchSpeedPrev = motorDirns[4]*motorFreqs[4]*(2*np.pi*(3/110)/400) #rad/s

        #yaw (400 steps per rotation)
        yawSpeedPrev = motorDirns[5]*motorFreqs[5]*(2*np.pi*0.02/400) #rad/s
    except:
        xSpeedPrev, ySpeedPrev, zSpeedPrev, rollSpeedPrev, pitchSpeedPrev, yawSpeedPrev =(0,0,0,0,0,0)

    return xSpeed, ySpeed, zSpeed, rollSpeed, pitchSpeed, yawSpeed, xSpeedPrev, ySpeedPrev, zSpeedPrev, rollSpeedPrev, pitchSpeedPrev, yawSpeedPrev
def definePosition(t, t_prev, step_dm):
    global yaxisPos_prev, zaxisPos_prev, gimbalHolderPos_prev, pitchAxisPos_prev, rollAxisPos_prev, endEffInterfacePos_prev
    tDiff = t-t_prev

    if manual:
        yaxisSpeed = yaxisSpeedPrev = 0.1
        zaxisSpeed = zaxisSpeedPrev = 0.2*np.cos(t)
        gimbalHolderSpeed = gimbalHolderSpeedPrev = 0.2*np.sin(t)
        pitchAxisSpeed = pitchAxisSpeedPrev = 0.5*np.sin(t)
        rollAxisSpeed = rollAxisSpeedPrev = np.sin(t)
        endEffInterfaceSpeed = endEffInterfaceSpeedPrev = 0.6
    else:
        yaxisSpeed, zaxisSpeed, gimbalHolderSpeed, pitchAxisSpeed, rollAxisSpeed, endEffInterfaceSpeed, yaxisSpeedPrev, zaxisSpeedPrev, gimbalHolderSpeedPrev, pitchAxisSpeedPrev, rollAxisSpeedPrev, endEffInterfaceSpeedPrev = getSpeeds(step_dm)

    #checking and adjusting for overshoot from 0.1s timestep (ofc only get overshoot from trajectory sims not manual)
    if int(t//dt_dm) != int(t_prev//dt_dm) and manual == False:
        dt_prev = math.ceil(t_prev * 10) / 10 - t_prev
        dt_post = t - math.floor(t * 10) / 10

        yaxisPos = yaxisPos_prev + dt_prev*yaxisSpeedPrev + dt_post*yaxisSpeed
        zaxisPos = zaxisPos_prev + dt_prev*zaxisSpeedPrev + dt_post*zaxisSpeed
        gimbalHolderPos = gimbalHolderPos_prev + dt_prev*gimbalHolderSpeedPrev + dt_post*gimbalHolderSpeed
        pitchAxisPos = pitchAxisPos_prev + dt_prev*pitchAxisSpeedPrev + dt_post*pitchAxisSpeed
        rollAxisPos = rollAxisPos_prev + dt_prev*rollAxisSpeedPrev + dt_post*rollAxisSpeed
        endEffInterfacePos = endEffInterfacePos_prev + dt_prev*endEffInterfaceSpeedPrev + dt_post*endEffInterfaceSpeed
    else:
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
def csvExport(ts, yaxisPosPlt, zaxisPosPlt, gimbalHolderPosPlt, pitchAxisPosPlt, rollAxisPosPlt, endEffInterfacePosPlt, endEffPosPlt, endEffOrientationPlt):
    #extract the x y z r p yw data
    endEff_xs, endEff_ys, endEff_zs = zip(*endEffPosPlt)
    x, y, z = list(endEff_xs), list(endEff_ys), list(endEff_zs)

    endEff_q1s, endEff_q2s, endEff_q3s, endEff_q0s = zip(*endEffOrientationPlt)
    q0, q1, q2, q3 = list(endEff_q0s), list(endEff_q1s), list(endEff_q2s), list(endEff_q3s)

    df = pd.DataFrame({0:ts, 1:x, 2:y, 3:z, 4:q0, 5:q1, 6:q2, 7:q3, 8:yaxisPosPlt, 9:zaxisPosPlt, 10:gimbalHolderPosPlt, 11:pitchAxisPosPlt, 12:rollAxisPosPlt, 13:endEffInterfacePosPlt})
    df.to_csv("VT_output_Traj/VT_Data_1.5-1.csv", header=False, index=False)

    print("Export successful!")
    return
def animateSystem():
    #initialise lists for plotting
    ts = []
    yaxisPosPlt = []
    zaxisPosPlt = []
    gimbalHolderPosPlt = []
    pitchAxisPosPlt = []
    rollAxisPosPlt = []
    endEffInterfacePosPlt = []
    endEffPosPlt = []
    endEffOrientationPlt = []

    #initialise simulation end scenario
    simEnd = False

    #total simulation time
    if manual:
        totalSimTime = tmaxManual
    else:
        #unknown until after simulation
        totalSimTime = 0

    #initialise limit switch activation checker
    #0 corresponds to no activation, 1-5 correspond to x, y, z, yaw, pitch (no limit switch on roll)
    limitSwitch = []
    limitUpperLower = []
    limitSwitchTimes = []
    limitUpperLowerArr = ["Upper", "Lower"]
    limitSwitchArr = ["x-axis", "y-axis", "z-axis", "yaw", "pitch"]
    limitHitting = np.array([False, False, False, False, False])
    
    #initialise collision checker
    colliding = False
    collisionTimes = []
    collisionLinks = []

    #get the simulation start time to ensure the simulation only runs for tmax amount of time
    simulationStart = time.time()
    t_prev = 0

    #begin the animation loop
    while True:
        simTime = time.time() - simulationStart

        step_dm = int(simTime//dt_dm) #the time-step of the dynamic model

        #determine if there is contact at this timestep
        contactGantryFrame = p.getContactPoints(gantry, gantry, endEffLinkIndex, gantryFrameLinkIndex)
        contactTargetFrame = p.getContactPoints(gantry, gantry, endEffLinkIndex, targetFrameLinkIndex)

        if (contactTargetFrame == () and contactGantryFrame == ()) and colliding:
            colliding = False
        elif contactGantryFrame != () and colliding == False:
            collisionTimes.append(simTime)
            collisionLinks.append("gantry frame")
            colliding = True
        elif contactTargetFrame != () and colliding == False:
            collisionTimes.append(simTime)
            collisionLinks.append("target frame")
            colliding = True

        #end the simulation at tmax or if a limit switch hit if in that mode, or if a collision occurs in that mode
        if simEnd or (limitSwitch != [] and limitBreakMode) or (contactGantryFrame != () and collisionBreakMode) or (contactTargetFrame != () and collisionBreakMode):
            #even if limit break mode off, will display which limits hit and when
            
            for i in range(len(limitSwitchTimes)):
                print(str(limitUpperLowerArr[limitUpperLower[i]]) + " " + str(limitSwitchArr[limitSwitch[i]]) + " hit at " + str(np.round(limitSwitchTimes[i],2)) + "s")

            for i in range(len(collisionTimes)):
                print("End effector collided with " + str(collisionLinks[i]) + " at " + str(np.round(collisionTimes[i],2)) + "s")

            csvExport(ts, yaxisPosPlt, zaxisPosPlt, gimbalHolderPosPlt, pitchAxisPosPlt, rollAxisPosPlt, endEffInterfacePosPlt, endEffPosPlt, endEffOrientationPlt)
            break

        #check if normal step or special 'final step' to make up the time to sim end perfectly
        if manual and simTime >= tmaxManual:
            simEnd = True
            yaxisPos, zaxisPos, gimbalHolderPos, pitchAxisPos, rollAxisPos, endEffInterfacePos = definePosition(tmaxManual, t_prev, step_dm)
            print("Simulation length: " + str(totalSimTime) + "s")
        elif manual == False and step_dm > len(motorControlArray) - 1:
            simEnd = True
            step_dm = len(motorControlArray) - 1
            totalSimTime = len(motorControlArray)*0.1
            yaxisPos, zaxisPos, gimbalHolderPos, pitchAxisPos, rollAxisPos, endEffInterfacePos = definePosition(totalSimTime, t_prev, step_dm)
            print("Simulation length: " + str(totalSimTime) + "s")
            print(simTime)
        else:
            #get new positions of each link
            yaxisPos, zaxisPos, gimbalHolderPos, pitchAxisPos, rollAxisPos, endEffInterfacePos = definePosition(simTime, t_prev, step_dm)

        #get relevant states of each joint relative to local origin
        yaxisCurrentPos = p.getJointState(gantry, yaxisPrismaticJointIndex)[0]
        zaxisCurrentPos = p.getJointState(gantry, zaxisPrismaticJointIndex)[0]
        gimbalHolderCurrentPos = p.getJointState(gantry, gimbalHolderPrismaticJointIndex)[0]
        pitchAxisCurrentPos = p.getJointState(gantry, pitchAxisRevoluteJointIndex)[0]
        rollAxisCurrentPos = p.getJointState(gantry, rollAxisRevoluteJointIndex)[0]
        endEffInterfaceCurrentPos = p.getJointState(gantry, endEffInterfaceRevoluteJointIndex)[0]

        #get state of end eff link relative to world origin (end eff interface used for testing)
        endEffCurrentPos = p.getLinkState(gantry, endEffInterfaceRevoluteJointIndex, computeForwardKinematics=True)[4]
        endEffCurrentOrientation = p.getLinkState(gantry, endEffInterfaceRevoluteJointIndex, computeForwardKinematics=True)[5]

        #append the current positions to the lists for plotting
        ts.append(simTime)
        yaxisPosPlt.append(yaxisCurrentPos)
        zaxisPosPlt.append(zaxisCurrentPos)
        gimbalHolderPosPlt.append(gimbalHolderCurrentPos)
        pitchAxisPosPlt.append(pitchAxisCurrentPos)
        rollAxisPosPlt.append(rollAxisCurrentPos)
        endEffInterfacePosPlt.append(endEffInterfaceCurrentPos)

        endEffPosPlt.append(endEffCurrentPos)
        endEffOrientationPlt.append(endEffCurrentOrientation)

        #move y-axis in x direction
        if yaxisPos >= yaxisJointUpperLimit and limitHitting[0] == False:
            #if joint is outside of limits, position will of course not update
            limitSwitch.append(0)
            limitSwitchTimes.append(simTime)
            limitUpperLower.append(0)
            limitHitting[0] = True
        elif yaxisPos <= yaxisJointLowerLimit and limitHitting[0] == False:
            limitSwitch.append(0)
            limitSwitchTimes.append(simTime)
            limitUpperLower.append(1)
            limitHitting[0] = True
        elif yaxisPos < yaxisJointUpperLimit and yaxisPos > yaxisJointLowerLimit:
            p.resetJointState(gantry, yaxisPrismaticJointIndex, yaxisPos)
            if limitHitting[0]:
                limitHitting[0] = False

        #move z-axis in y direction
        if zaxisPos >= zaxisJointUpperLimit and limitHitting[1] == False:
            #if joint is outside of limits, position will of course not update
            limitSwitch.append(1)
            limitSwitchTimes.append(simTime)
            limitUpperLower.append(0)
            limitHitting[1] = True
        elif zaxisPos <= zaxisJointLowerLimit and limitHitting[1] == False:
            limitSwitch.append(1)
            limitSwitchTimes.append(simTime)
            limitUpperLower.append(1)
            limitHitting[1] = True
        elif zaxisPos < zaxisJointUpperLimit and zaxisPos > zaxisJointLowerLimit:
            p.resetJointState(gantry, zaxisPrismaticJointIndex, zaxisPos)
            if limitHitting[1]:
                limitHitting[1] = False

        #move gimbal in z direction
        if gimbalHolderPos >= gimbalHolderJointUpperLimit and limitHitting[2] == False:
            #if joint is outside of limits, position will of course not update
            limitSwitch.append(2)
            limitSwitchTimes.append(simTime)
            limitUpperLower.append(0)
            limitHitting[2] = True
        elif gimbalHolderPos <= gimbalHolderJointLowerLimit and limitHitting[2] == False:
            limitSwitch.append(2)
            limitSwitchTimes.append(simTime)
            limitUpperLower.append(1)
            limitHitting[2] = True
        elif gimbalHolderPos < gimbalHolderJointUpperLimit and gimbalHolderPos > gimbalHolderJointLowerLimit:
            p.resetJointState(gantry, gimbalHolderPrismaticJointIndex, gimbalHolderPos)
            if limitHitting[2]:
                limitHitting[2] = False

        #rotate the pitch axis in yaw
        if pitchAxisPos >= pitchAxisJointUpperLimit and limitHitting[3] == False:
            #if joint is outside of limits, position will of course not update
            limitSwitch.append(3)
            limitSwitchTimes.append(simTime)
            limitUpperLower.append(0)
            limitHitting[3] = True
        elif pitchAxisPos <= pitchAxisJointLowerLimit and limitHitting[3] == False:
            limitSwitch.append(3)
            limitSwitchTimes.append(simTime)
            limitUpperLower.append(1)
            limitHitting[3] = True
        elif pitchAxisPos < pitchAxisJointUpperLimit and pitchAxisPos > pitchAxisJointLowerLimit:
            p.resetJointState(gantry, pitchAxisRevoluteJointIndex, pitchAxisPos)
            if limitHitting[3]:
                limitHitting[3] = False

        #rotate the roll axis in pitch
        if rollAxisPos >= rollAxisJointUpperLimit and limitHitting[4] == False:
            #if joint is outside of limits, position will of course not update
            limitSwitch.append(4)
            limitSwitchTimes.append(simTime)
            limitUpperLower.append(0)
            limitHitting[4] = True
        elif rollAxisPos <= rollAxisJointLowerLimit and limitHitting[4] == False:
            limitSwitch.append(4)
            limitSwitchTimes.append(simTime)
            limitUpperLower.append(1)
            limitHitting[4] = True
        elif rollAxisPos < rollAxisJointUpperLimit and rollAxisPos > rollAxisJointLowerLimit:
            p.resetJointState(gantry, rollAxisRevoluteJointIndex, rollAxisPos)
            if limitHitting[4]:
                limitHitting[4] = False

        #rotate the end effector interface in roll
        p.resetJointState(gantry, endEffInterfaceRevoluteJointIndex, endEffInterfacePos)

        t_prev = simTime
        p.stepSimulation() #maybe take out? if results not looking good then try
        #max refresh rate of pybullet, not necessarily same timestep as is used for plotting but of course same function
        time.sleep(1/240)
def getJSON():
    #very similar to animate function but without the plotting stuff, and with the JSON saving

    #list to append all of the frames to
    all_frames = []

    #initialise simulation end scenario
    simEnd = False

    #total simulation time
    if manual:
        totalSimTime = tmaxManual
    else:
        #unknown until after simulation
        totalSimTime = 0

    #initialise limit switch activation checker
    #0 corresponds to no activation, 1-5 correspond to x, y, z, yaw, pitch (no limit switch on roll)
    limitSwitch = []
    limitUpperLower = []
    limitSwitchTimes = []
    limitUpperLowerArr = ["Upper", "Lower"]
    limitSwitchArr = ["x-axis", "y-axis", "z-axis", "yaw", "pitch"]
    limitHitting = np.array([False, False, False, False, False])
    
    #initialise collision checker
    colliding = False
    collisionTimes = []
    collisionLinks = []

    #get the simulation start time to ensure the simulation only runs for tmax amount of time
    simulationStart = time.time()
    t_prev = 0

    #FOR JSON COLLECTING
    frame = 0

    #begin the animation loop
    while True:
        simTime = time.time() - simulationStart

        step_dm = int(simTime//dt_dm) #the time-step of the dynamic model

        #determine if there is contact at this timestep
        contactGantryFrame = p.getContactPoints(gantry, gantry, endEffLinkIndex, gantryFrameLinkIndex)
        contactTargetFrame = p.getContactPoints(gantry, gantry, endEffLinkIndex, targetFrameLinkIndex)

        if (contactTargetFrame == () and contactGantryFrame == ()) and colliding:
            colliding = False
        elif contactGantryFrame != () and colliding == False:
            collisionTimes.append(simTime)
            collisionLinks.append("gantry frame")
            colliding = True
        elif contactTargetFrame != () and colliding == False:
            collisionTimes.append(simTime)
            collisionLinks.append("target frame")
            colliding = True

        #end the simulation at tmax or if a limit switch hit if in that mode, or if a collision occurs in that mode
        if simEnd or (limitSwitch != [] and limitBreakMode) or (contactGantryFrame != () and collisionBreakMode) or (contactTargetFrame != () and collisionBreakMode):
            #even if limit break mode off, will display which limits hit and when
            
            for i in range(len(limitSwitchTimes)):
                print(str(limitUpperLowerArr[limitUpperLower[i]]) + " " + str(limitSwitchArr[limitSwitch[i]]) + " hit at " + str(np.round(limitSwitchTimes[i],2)))

            for i in range(len(collisionTimes)):
                print("End effector collided with " + str(collisionLinks[i]) + " at " + str(np.round(collisionTimes[i],2)))

            break

        #check if normal step or special 'final step' to make up the time to sim end perfectly
        if manual and simTime >= tmaxManual:
            simEnd = True
            yaxisPos, zaxisPos, gimbalHolderPos, pitchAxisPos, rollAxisPos, endEffInterfacePos = definePosition(tmaxManual, t_prev, step_dm)
            print("Simulation length: " + str(totalSimTime) + "s")
        elif manual == False and step_dm > len(motorControlArray) - 1:
            simEnd = True
            step_dm = len(motorControlArray) - 1
            totalSimTime = len(motorControlArray)*0.1
            yaxisPos, zaxisPos, gimbalHolderPos, pitchAxisPos, rollAxisPos, endEffInterfacePos = definePosition(totalSimTime, t_prev, step_dm)
            print("Simulation length: " + str(totalSimTime) + "s")
            print(simTime)
        else:
            #get new positions of each link
            yaxisPos, zaxisPos, gimbalHolderPos, pitchAxisPos, rollAxisPos, endEffInterfacePos = definePosition(simTime, t_prev, step_dm)

        #move y-axis in x direction
        if yaxisPos >= yaxisJointUpperLimit and limitHitting[0] == False:
            #if joint is outside of limits, position will of course not update
            limitSwitch.append(0)
            limitSwitchTimes.append(simTime)
            limitUpperLower.append(0)
            limitHitting[0] = True
        elif yaxisPos <= yaxisJointLowerLimit and limitHitting[0] == False:
            limitSwitch.append(0)
            limitSwitchTimes.append(simTime)
            limitUpperLower.append(1)
            limitHitting[0] = True
        elif yaxisPos < yaxisJointUpperLimit and yaxisPos > yaxisJointLowerLimit:
            p.resetJointState(gantry, yaxisPrismaticJointIndex, yaxisPos)
            if limitHitting[0]:
                limitHitting[0] = False

        #move z-axis in y direction
        if zaxisPos >= zaxisJointUpperLimit and limitHitting[1] == False:
            #if joint is outside of limits, position will of course not update
            limitSwitch.append(1)
            limitSwitchTimes.append(simTime)
            limitUpperLower.append(0)
            limitHitting[1] = True
        elif zaxisPos <= zaxisJointLowerLimit and limitHitting[1] == False:
            limitSwitch.append(1)
            limitSwitchTimes.append(simTime)
            limitUpperLower.append(1)
            limitHitting[1] = True
        elif zaxisPos < zaxisJointUpperLimit and zaxisPos > zaxisJointLowerLimit:
            p.resetJointState(gantry, zaxisPrismaticJointIndex, zaxisPos)
            if limitHitting[1]:
                limitHitting[1] = False

        #move gimbal in z direction
        if gimbalHolderPos >= gimbalHolderJointUpperLimit and limitHitting[2] == False:
            #if joint is outside of limits, position will of course not update
            limitSwitch.append(2)
            limitSwitchTimes.append(simTime)
            limitUpperLower.append(0)
            limitHitting[2] = True
        elif gimbalHolderPos <= gimbalHolderJointLowerLimit and limitHitting[2] == False:
            limitSwitch.append(2)
            limitSwitchTimes.append(simTime)
            limitUpperLower.append(1)
            limitHitting[2] = True
        elif gimbalHolderPos < gimbalHolderJointUpperLimit and gimbalHolderPos > gimbalHolderJointLowerLimit:
            p.resetJointState(gantry, gimbalHolderPrismaticJointIndex, gimbalHolderPos)
            if limitHitting[2]:
                limitHitting[2] = False

        #rotate the pitch axis in yaw
        if pitchAxisPos >= pitchAxisJointUpperLimit and limitHitting[3] == False:
            #if joint is outside of limits, position will of course not update
            limitSwitch.append(3)
            limitSwitchTimes.append(simTime)
            limitUpperLower.append(0)
            limitHitting[3] = True
        elif pitchAxisPos <= pitchAxisJointLowerLimit and limitHitting[3] == False:
            limitSwitch.append(3)
            limitSwitchTimes.append(simTime)
            limitUpperLower.append(1)
            limitHitting[3] = True
        elif pitchAxisPos < pitchAxisJointUpperLimit and pitchAxisPos > pitchAxisJointLowerLimit:
            p.resetJointState(gantry, pitchAxisRevoluteJointIndex, pitchAxisPos)
            if limitHitting[3]:
                limitHitting[3] = False

        #rotate the roll axis in pitch
        if rollAxisPos >= rollAxisJointUpperLimit and limitHitting[4] == False:
            #if joint is outside of limits, position will of course not update
            limitSwitch.append(4)
            limitSwitchTimes.append(simTime)
            limitUpperLower.append(0)
            limitHitting[4] = True
        elif rollAxisPos <= rollAxisJointLowerLimit and limitHitting[4] == False:
            limitSwitch.append(4)
            limitSwitchTimes.append(simTime)
            limitUpperLower.append(1)
            limitHitting[4] = True
        elif rollAxisPos < rollAxisJointUpperLimit and rollAxisPos > rollAxisJointLowerLimit:
            p.resetJointState(gantry, rollAxisRevoluteJointIndex, rollAxisPos)
            if limitHitting[4]:
                limitHitting[4] = False

        #rotate the end effector interface in roll
        p.resetJointState(gantry, endEffInterfaceRevoluteJointIndex, endEffInterfacePos)

        t_prev = simTime

        #JSON STUFF
        frame_data = {"frame": frame, "links": []}

        base_pos, base_orn = p.getBasePositionAndOrientation(gantry)
        frame_data["links"].append({"name": "base_link", "position": base_pos, "orientation": base_orn})

        for link_index in range(p.getNumJoints(gantry)):
            link_state = p.getLinkState(gantry, link_index, computeForwardKinematics=True)
            pos, orn = link_state[4], link_state[5]
            link_name = p.getJointInfo(gantry, link_index)[12].decode("utf-8")
            frame_data["links"].append({"name": link_name, "position": pos, "orientation": orn})

        all_frames.append(frame_data)
        frame += 1

        #END OF JSON STUFF

        p.stepSimulation() #maybe take out? if results not looking good then try
        #max refresh rate of pybullet, not necessarily same timestep as is used for plotting but of course same function
        time.sleep(1/240)

    return all_frames



###############################################
#             PYBULLET INITIATION             #
###############################################
gantry = pybulletInit()

#assign the correct joint number to each joint
yaxisPrismaticJointIndex = 1
zaxisPrismaticJointIndex = 2
gimbalHolderPrismaticJointIndex = 3
pitchAxisRevoluteJointIndex = 5
rollAxisRevoluteJointIndex = 6
endEffInterfaceRevoluteJointIndex = 7

#assign link indices to links that need to be checked for collision
gantryFrameLinkIndex = 0
endEffLinkIndex = 8
targetFrameLinkIndex = 9

#define limits of each joint
yaxisJointLowerLimit, yaxisJointUpperLimit, zaxisJointLowerLimit, zaxisJointUpperLimit, gimbalHolderJointLowerLimit, gimbalHolderJointUpperLimit, pitchAxisJointLowerLimit, pitchAxisJointUpperLimit, rollAxisJointLowerLimit, rollAxisJointUpperLimit = getJointLims()

#define paramters used to calculate motor frequency
PREnormal = np.array([1,8,64,256,1024])
PRE2 = np.array([1,8,32,64,128,256,1024]) #motor 2 is pitch
Fclock = np.array([16e6,16e6,16e6,8e6,8e6,16e6]) #x y z r p yw



###############################################
#             VARIABLE DEFINITIONS            #
###############################################

#store the motor controls in an array from the csv
df = pd.read_csv("VT_Motor_Control_Traj/virtualTwinData_TC1.5-1.csv", header=None)
motorControlArray = df.to_numpy()
print(motorControlArray)

#define time the same way it was done in the dynamic model to make future integration easier
dt_dm = 0.1 #dynamic model dt (dt over which the velocity is changed)
tmaxManual = 5 #seconds (assuming a constant timestep)

#define the intitial position of each link (in the motor frame)
yaxisPos_prev = 0.1352 #m (x)
zaxisPos_prev = -0.7161 #m NEGATIVE (y)
gimbalHolderPos_prev = 0.7773 #m (z)
pitchAxisPos_prev = -1*np.deg2rad(-30) #rad NEGATIVE (yw)
rollAxisPos_prev = np.deg2rad(40) #rad (p)
endEffInterfacePos_prev = np.deg2rad(-50) #rad NOT NEGATIVE (r)

#define the modes of the simulation
animate = True #do we want to animate (control), or move with mouse
manual = False #following a trajectory? Or manual speed control for debugging
blender = False #when set to true exports an animation to blender
limitBreakMode = False #when set to true, hitting a limit switch will end the simulation
collisionBreakMode = False #when set to true, a collision will end the simulation



###############################################
#                  ANIMATION                  #
###############################################

#set the position to the start position before the simulation starts
setInitialPosition()

if animate:
    #call the animate function here
    animateSystem()

elif blender:
    all_frames = getJSON()

    with open("animation.json", "w") as f:
        json.dump(all_frames, f, indent=2)

else:
    p.setRealTimeSimulation(1)

    #Draw a line at each joint origin
    # for joint_index in range(9):
    #     link_state = p.getLinkState(gantry, joint_index)
    #     world_position = link_state[0]

    #     p.addUserDebugText("O", world_position, textSize=1, textColorRGB=[1, 0, 0])
    #     p.addUserDebugLine(world_position, [world_position[0], world_position[1], world_position[2] + 0.1], [1, 0, 0], 2)
        
    #label the axis
    p.addUserDebugText("X", (0.9,-0.1,0), textSize = 2, textColorRGB=[1,0,0])
    p.addUserDebugText("Y", (-0.1,0.9,0), textSize = 2, textColorRGB=[0,1,0])
    p.addUserDebugText("Z", (-0.1,-0.1,0.9), textSize = 2, textColorRGB=[0,0,1])

    while True:
        pass