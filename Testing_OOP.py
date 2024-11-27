from SC_Architecture import Thruster, ReactionWheel, SpaceCraft
import numpy as np

T1 = Thruster("Thruster1", np.array([[0,1,1]]), np.array([[2,3,1]])) # name, position, orientation
T2 = Thruster("Thruster2", np.array([[1,2,3]]), np.array([[6,1,1]]))
T3 = Thruster("BIGTHRUSTER", np.array([[1,2,3]]), np.array([[6,1,1]]))

RW1 = ReactionWheel("ReactionWheel1", np.array([[0,0,0]]), np.array([[1,4,1]])) # name, position, orientation
RW2 = ReactionWheel("ReactionWheel2", np.array([[1,1,1]]), np.array([[0,3,1]]))

CubeSat = SpaceCraft("RAYWATCH", 10, np.array([[1,0,0],[0,1,0],[0,0,1]]), [T1, T2, T3], [RW1, RW2]) # name, mass, I, Thrusters, RWs

print(CubeSat)