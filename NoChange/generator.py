import numpy as np

def normalize(x):
    return ((x+np.pi)%(2*np.pi)) - np.pi

# def get_trajectories(target):

#     # Generate Polynomial Equation from Desired Target
#     desired_z = np.array([4.698, 3.23808163, 2.23808163, 1.71987755, 0.7, 0.73718367, 1.94665306, 2.29])
#     x = np.linspace(target[0]-2, target[0]+2, 8)
#     a,b,c = np.polyfit(x,desired_z,2)

#     # Calculate the angle from positive x-axis
#     theta = normalize(np.arctan2(target[1],target[0]))

#     # Calculate how much y will be shifted
#     xt = abs(target[0])
#     y_shift = xt*np.sin(theta)

#     shift = target[1] - y_shift

#     # Generate x,y,z based on desired trajectory length
#     x = np.sort(x, kind="quicksort")[::-1]
#     y = abs(x)*np.sin(theta) + shift
#     z = desired_z

#     orientation_targets =  np.array([-0.6,
#                                     -0.8,
#                                     -1.6,
#                                     -1.8,
#                                     -2.073,
#                                     -2.073, 
#                                     -2.966,
#                                     -2.874])

#     return (np.array([x,y,z]).T,orientation_targets), theta

def get_trajectories():
    position_targets = np.array([[10.194,0,4.2],
                                    [10.568,0,2.41],
                                    [9.817,0,0.942],
                                    [8.22,0,0.75],
                                    [6.64,0,1.104],
                                    [6.65,0,2.08]])

    orientation_targets =  np.array([-0.6,
                                        -0.8,
                                        -1.839,
                                        -2.073, 
                                        -2.966,
                                        -2.874])
    theta = 0
    return (position_targets, orientation_targets), theta