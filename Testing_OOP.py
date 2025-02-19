from SC_Architecture import Thruster, ReactionWheel, SpaceCraft
import numpy as np


az = 30
el = 20
Xs = 3
Ys = 2
Zs = 1
F_max = 10

T1 = Thruster("Thruster1", np.array([[Xs/2,Ys/2,Zs/2]]), np.array([[az, el]]), F_max) # name, position, orientation (az, el)
T2 = Thruster("Thruster2", np.array([[Xs/2,-Ys/2,Zs/2]]), np.array([[-az, el]]), F_max)
T3 = Thruster("Thruster3", np.array([[-Xs/2,-Ys/2,Zs/2]]), np.array([[az, 180-el]]), F_max)
T4 = Thruster("Thruster4", np.array([[-Xs/2,Ys/2,Zs/2]]), np.array([[-az, 180-el]]), F_max)
T5 = Thruster("Thruster5", np.array([[Xs/2,Ys/2,-Zs/2]]), np.array([[az, -el]]), F_max) # name, position, orientation
T6 = Thruster("Thruster6", np.array([[Xs/2,-Ys/2,-Zs/2]]), np.array([[-az, -el]]), F_max)
T7 = Thruster("Thruster7", np.array([[-Xs/2,-Ys/2,-Zs/2]]), np.array([[az, -(180-el)]]), F_max)
T8 = Thruster("Thruster8", np.array([[-Xs/2,Ys/2,-Zs/2]]), np.array([[-az, -(180-el)]]), F_max)
Thrusters = [T1,T2,T3,T4,T5,T6,T7,T8]

RW1 = ReactionWheel("ReactionWheel1", np.array([[0,0,0]]), np.array([[1,4]]), 10) # name, position, orientation
RW2 = ReactionWheel("ReactionWheel2", np.array([[1,1,1]]), np.array([[0,3]]), 10)
RWs = [RW1,RW2]

CubeSat = SpaceCraft("RAYWATCH", 10, np.array([[1,0,0],[0,1,0],[0,0,1]]), np.array([Xs, Ys, Zs]), Thrusters, RWs) # name, mass, I, Thrusters, RWs, SC_geometry

optimal_thrusters_dic = CubeSat.thruster_states_dic

print("Fx: {}.".format(optimal_thrusters_dic["Fx"]))
print("-Fx: {}.".format(optimal_thrusters_dic["-Fx"]))
print("Fy: {}.".format(optimal_thrusters_dic["Fy"]))
print("-Fy: {}.".format(optimal_thrusters_dic["-Fy"]))
print("Fz: {}.".format(optimal_thrusters_dic["Fz"]))
print("-Fz: {}.".format(optimal_thrusters_dic["-Fz"]))
print("Tx: {}.".format(optimal_thrusters_dic["Tx"]))
print("-Tx: {}.".format(optimal_thrusters_dic["-Tx"]))
print("Ty: {}.".format(optimal_thrusters_dic["Ty"]))
print("-Ty: {}.".format(optimal_thrusters_dic["-Ty"]))
print("Tz: {}.".format(optimal_thrusters_dic["Tz"]))
print("-Tz: {}.".format(optimal_thrusters_dic["-Tz"]))


CubeSat.DrawSpaceCraft(DRAW_THRUSTERS=1)
print(CubeSat)

