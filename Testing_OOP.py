from SC_Architecture import Thruster, ReactionWheel, SpaceCraft
import numpy as np

az = 20
el = 0
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
#T3 = Thruster("BIGTHRUSTER", np.array([[1,2,3]]), np.array([[6,1,1]]))

RW1 = ReactionWheel("ReactionWheel1", np.array([[0,0,0]]), np.array([[1,4]]), 10) # name, position, orientation
#RW2 = ReactionWheel("ReactionWheel2", np.array([[1,1,1]]), np.array([[0,3,1]]))
CubeSat = SpaceCraft("RAYWATCH", 10, np.array([[1,0,0],[0,1,0],[0,0,1]]), np.array([Xs, Ys, Zs]), [T1,T2,T3,T4,T5,T6,T7,T8], [RW1]) # name, mass, I, Thrusters, RWs, SC_geometry

from itertools import combinations

def find_min_thruster_combination(T):
    """
    Finds the minimum number of thrusters that satisfy:
    - Only force in X (Fy = 0, Fz = 0, τx = 0, τy = 0, τz = 0).
    - Fx must be positive.
    - Thrusters are ON (1) or OFF (0).
    
    Args:
        T (np.ndarray): ThrustVectorMatrix (n x 6), where columns are [Fx, Fy, Fz, τx, τy, τz]
    
    Returns:
        np.ndarray: Binary vector (1 = thruster ON, 0 = OFF).
    """
    num_thrusters = T.shape[0]
    
    # Try increasing numbers of thrusters, starting with the smallest sets
    for r in range(1, num_thrusters + 1):
        for thruster_indices in combinations(range(num_thrusters), r):
            thruster_states = np.zeros(num_thrusters)  # All off
            thruster_states[list(thruster_indices)] = 1  # Turn on selected thrusters

            # Compute total force & torque
            resultant_force_torque = np.sum(T * thruster_states[:, None], axis=0)
            Fx, Fy, Fz, τx, τy, τz = resultant_force_torque

            # Check if it meets the criteria
            if Fx > 0 and Fy == 0 and Fz == 0 and τx == 0 and τy == 0 and τz == 0:
                return thruster_states  # Return the first valid minimum-thruster solution
    
    print("No valid thruster configuration found.")
    return None

# Example usage
optimal_thrusters = find_min_thruster_combination(CubeSat.ThrustVectorMatrix)

if optimal_thrusters is not None:
    print("Optimal thrusters to activate:", optimal_thrusters)
else:
    print("No valid thruster configuration found.")


print(CubeSat)
print(CubeSat.ThrustVectorMatrix)
CubeSat.DrawSpaceCraft(DRAW_THRUSTERS=1)


