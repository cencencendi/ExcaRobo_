import numpy as np

joints_targets = np.array([[-0.7,1.4,0.4],
                                        [-0.5,0.7,0.4],
                                        [-0.3,0.6,0.5],
                                        [-0.144,0.59,1.47],
                                        [-0.257,1.17,1.19],
                                        [-0.294,1.437,1.823],
                                        [-0.444,1.458,1.859],
                                        [-0.444,1.46,0.276]])

position_targets = np.array([[8.12,0,4.07],
                                    [10.194,0,4.2],
                                    [10.568,0,2.41],
                                    [9.817,0,0.942],
                                    [8.22,0,0.75],
                                    [6.64,0,1.104],
                                    [6.65,0,2.08],
                                    [9.19,4,1.71]])

orientation_targets =  np.array([-1.1,
                                        -0.6,
                                        -0.8,
                                        -1.839,
                                        -2.073, 
                                        -2.966,
                                        -2.874,
                                        -0.792])

initial_position = [position_targets[0], orientation_targets[0]]
digging_operation = [position_targets[1:-1], orientation_targets[1:-1]]
release = [position_targets[-1], orientation_targets[-1]]