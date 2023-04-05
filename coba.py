import pybullet as p
import time
import pybullet_data
import matplotlib.pyplot as plt
import numpy as np
physicsClient = p.connect(p.GUI)#or p.DIRECT for non-graphical version
p.setAdditionalSearchPath(pybullet_data.getDataPath()) #optionally
p.setGravity(0,0,-9.8)
 
planeId = p.loadURDF("plane.urdf")
 
startPos = [0, 0, 1.4054411813121799]
startOrientation = p.getQuaternionFromEuler([0,0,0])
boxId = p.loadURDF("aba_excavator/excavator.urdf",startPos, startOrientation)
o,o1 = [], []

pose_target = np.array([[-0.7,1.4,0.4],
                        [-0.5,0.7,0.4],
                        [-0.3,0.6,0.5],
                           [-0.144,0.59,1.47],
                           [-0.257,1.17,1.19],
                           [-0.294,1.437,1.823],
                           [-0.444,1.458,1.859],
                           [-0.444,1.46,0.276]])

def _get_joint_state():
        theta0, theta1, theta2 = p.getJointStates(boxId, [2,3,4])
        theta_now = normalize(np.array([theta0[0], theta1[0], theta2[0]]))
        theta_dot_now = np.array([theta0[1], theta1[1], theta2[1]])
        return theta_now, theta_dot_now

def normalize(x):
    return ((x+np.pi)%(2*np.pi)) - np.pi

# for pose in pose_target:
#     print(pose)
last_ori = 0
for i in range(5000):
    p.setJointMotorControl2(boxId, 1 , p.VELOCITY_CONTROL, targetVelocity = 0, force=10000)
    p.setJointMotorControl2(boxId, 2 , p.VELOCITY_CONTROL, targetVelocity = -0.3, force= 250_000)
    p.setJointMotorControl2(boxId, 3 , p.VELOCITY_CONTROL, targetVelocity = -0.3, force= 250_000)
    p.setJointMotorControl2(boxId, 4 , p.VELOCITY_CONTROL, targetVelocity = -0.3, force= 250_000)
    (linkWorldPosition,
            linkWorldOrientation,
            localInertialFramePosition,
            localInertialFrameOrientation,
            worldLinkFramePosition,
            worldLinkFrameOrientation,
            worldLinkLinearVelocity,
            worldLinkAngularVelocity) = p.getLinkState(boxId, 4, computeLinkVelocity=1, computeForwardKinematics=1)
    # print(linkWorldPosition)    
    p.stepSimulation()
    time.sleep(1.0/240.0)
#     swing = p.getJointState(boxId, 1)[0]
    theta_now , _= _get_joint_state()
    orientation_now = normalize(-sum(theta_now))
    orientation_velocity = (last_ori-orientation_now)/(1.0/240.0)
#     # theta1 = p.getJointState(boxId,2)
#     # theta2 = p.getJointState(boxId,3)
#     # theta3 = p.getJointState(boxId,4)
#     # print(linkWorldPosition)
#     # ori = normalize(-(theta1[0]+theta2[0]+theta3[0]))
#     # orientation = p.getEulerFromQuaternion(orientation)
#     # print(orientation[1], ori)
#     # o.append(np.array(linkWorldPosition))
#     # o1.append(ori)
    last_ori = orientation_now
    print(orientation_velocity)
    print(worldLinkLinearVelocity)
#     print(swing)
# while True:
#     theta = p.getJointState(boxId, 1)[0]
#     error = (np.pi/2 - theta)
#     vel = 0.5*error
#     p.setJointMotorControl2(boxId, 1 , p.VELOCITY_CONTROL, targetVelocity = vel, force=50_000)
#     p.setJointMotorControl2(boxId, 2 , p.VELOCITY_CONTROL, targetVelocity = 0, force=250_000)
#     p.setJointMotorControl2(boxId, 3 , p.VELOCITY_CONTROL, targetVelocity = 0, force=250_000)
#     p.setJointMotorControl2(boxId, 4 , p.VELOCITY_CONTROL, targetVelocity = 0, force=250_000)
#     p.stepSimulation()
#     time.sleep(1.0/240.)
#     (linkWorldPosition, orientation, *_) = p.getLinkState(boxId,4, computeLinkVelocity=1, computeForwardKinematics=1)
#     # print(linkWorldPosition)    
#     p.stepSimulation()
#     time.sleep(1.0/240.)
#     swing = p.getJointState(boxId, 1)[0]
#     theta_now , _= _get_joint_state()
#     print(error)
#     thetat = np.arctan2(linkWorldPosition[1],linkWorldPosition[0])

#     if(np.pi/2 - theta)<5e-3:
#         p.setJointMotorControl2(boxId, 1 , p.VELOCITY_CONTROL, targetVelocity = 0, force=50_000)
#         p.setJointMotorControl2(boxId, 2 , p.VELOCITY_CONTROL, targetVelocity = 0, force=250_000)
#         p.setJointMotorControl2(boxId, 3 , p.VELOCITY_CONTROL, targetVelocity = 0, force=250_000)
#         p.setJointMotorControl2(boxId, 4 , p.VELOCITY_CONTROL, targetVelocity = 0, force=250_000)
#         p.stepSimulation()
#         time.sleep(1.0/240.)
#         print(np.pi/2-theta)
#         swing = p.getJointState(boxId, 1)[0]
#         break


# print(np.array(linkWorldPosition))
# print(linkWorldPosition[0]/np.cos(swing))
# print(linkWorldPosition[1]/np.sin(swing))
p.disconnect()

# o = np.array(o)
# plt.plot(o[:,0],o[:,2])
# plt.plot(list(range(1000)),np.array(o1))
# plt.show()
