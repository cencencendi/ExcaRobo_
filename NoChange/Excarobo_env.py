import gym
import time
import numpy as np
import pybullet as p
from gym import spaces
import pybullet_data
import random

class ExcaRobo(gym.Env):
    def __init__(self, render_mode=None):
        super(ExcaRobo, self).__init__()
        if render_mode=='human':
            physicsClient = p.connect(p.GUI)#or p.DIRECT for non-graphical version
        else:
            physicsClient = p.connect(p.DIRECT)#or p.DIRECT for non-graphical version

        self.MAX_EPISODE = 3_000
        self.dt = 1.0/240.0
        self.max_theta = [1.03, 1.51, 3.14]    
        self.min_theta = [-0.954, -0.1214, -0.32]
        self.observation_space = spaces.Box(low =-np.inf, high = np.inf, shape= (25,), dtype=np.float32)
        self.action_space = spaces.Box(low = -0.3, high = 0.3, shape=(3,), dtype=np.float32)
        self.steps_left = np.copy(self.MAX_EPISODE)
        
        self.start_simulation()

    def step(self, action):

        p.setJointMotorControl2(self.boxId, 2 , p.VELOCITY_CONTROL, targetVelocity = action[0], force= 250_000)
        p.setJointMotorControl2(self.boxId, 3 , p.VELOCITY_CONTROL, targetVelocity = action[1], force= 250_000)
        p.setJointMotorControl2(self.boxId, 4 , p.VELOCITY_CONTROL, targetVelocity = action[2], force= 250_000)

        #Update Simulations
        p.stepSimulation()
        time.sleep(self.dt)

        #Orientation Error
        self.theta_now, self.theta_dot_now = self._get_joint_state()
        self.orientation_now = self.normalize(-sum(self.theta_now))

        orientation_error = self.rotmat2theta(
            self.rot_mat(self.orientation_target)@self.rot_mat(self.orientation_now).T
        )
        desired_orientation_velocity = 5*orientation_error

        self.orientation_velocity = (self.orientation_now-self.orientation_last)/self.dt

        #Position error
        theta_swing = p.getJointState(self.boxId, 1)[0]
        position_now, self.link_velocity = self._get_link_state()
        xyzw_world = np.concatenate((position_now,[1]))

        homogeneous_transformation_swing = self.homogeneous_transformation(theta_swing)

        xyzw_swing = np.linalg.inv(homogeneous_transformation_swing)@xyzw_world

        self.position_now = xyzw_swing[:-1]

        vec = np.array(self.position_now) - self.position_target
        desired_linear_velocity = -5*vec

        #Reward Calculation
        reward_dist = 4*np.exp(-np.linalg.norm(desired_linear_velocity-self.link_velocity))
        reward_orientation = -0.02*(desired_orientation_velocity-self.orientation_velocity)**2
        reward_ctrl = -0.0075*np.linalg.norm(action)

        reward = reward_dist + reward_ctrl + reward_orientation

        #Update State and Observation Space
        self.new_obs = self._get_obs(action = action, 
                                     desired_orientation_velocity = desired_orientation_velocity, 
                                     desired_linear_velocity = desired_linear_velocity, 
                                     error = vec, 
                                     orientation_error = orientation_error)

        #Set the condition
        # if np.any(self.theta_now > np.array(self.max_theta)) or np.any(self.theta_now < np.array(self.min_theta)):
        #     print(f"theta_now: {self.theta_now}\n")
        #     done = True
        #     punishment = -1000
        #     self.reward = reward+punishment
        if np.all(abs(vec)<8e-2) and self.steps_left<2500:
            self.reward = reward
            done = True
        else:
            done = bool(self.steps_left<0)
            self.steps_left -= 1
            self.reward = reward

        self.orientation_last = self.orientation_now
        self.last_act = action
        self.cur_done = done
        
        return self.new_obs, self.reward, done, {}

    def start_simulation(self):
        p.setAdditionalSearchPath(pybullet_data.getDataPath()) #optionally

        ## Setup Physics
        p.setGravity(0,0,-9.8)

        ## Load Plane
        planeId = p.loadURDF("plane.urdf")

        ## Load Robot
        startPos = [0,0,1.4054411813121799]
        startOrientation = p.getQuaternionFromEuler([0,0,0])
        self.boxId = p.loadURDF("../aba_excavator/excavator.urdf",startPos, startOrientation)

    def reset(self):
    
        #Reset Simulation
        p.resetSimulation()
        self.start_simulation()
        
        # start_position = np.array([-0.7,1.4,0])
        # vel = np.zeros(3)

        # while True:
        #     theta_now, _ = self._get_joint_state()
        #     for i in range(3):
        #         err = self.rotmat2theta(self.rot_mat(start_position[i])@self.rot_mat(theta_now[i]).T)
        #         vel[i] = 2*err

        #     p.setJointMotorControl2(self.boxId, 2 , p.VELOCITY_CONTROL, targetVelocity = vel[0], force= 150_000)
        #     p.setJointMotorControl2(self.boxId, 3 , p.VELOCITY_CONTROL, targetVelocity = vel[1], force= 150_000)
        #     p.setJointMotorControl2(self.boxId, 4 , p.VELOCITY_CONTROL, targetVelocity = vel[2], force= 150_000)

        #     #Update Simulations
        #     p.stepSimulation()
        #     time.sleep(self.dt)

        #     if np.all(abs(theta_now-start_position)<1e-1):
        #         p.setJointMotorControl2(self.boxId, 2 , p.VELOCITY_CONTROL, targetVelocity = 0, force= 150_000)
        #         p.setJointMotorControl2(self.boxId, 3 , p.VELOCITY_CONTROL, targetVelocity = 0, force= 150_000)
        #         p.setJointMotorControl2(self.boxId, 4 , p.VELOCITY_CONTROL, targetVelocity = 0, force= 150_000)
        #         p.stepSimulation()
        #         time.sleep(self.dt)
        #         break

        #Get Joint State
        self.theta_now, self.theta_dot_now = self._get_joint_state()
        self.orientation_last = self.normalize(-sum(self.theta_now))
        self.orientation_now = self.normalize(-sum(self.theta_now))
        self.orientation_velocity = (self.orientation_now-self.orientation_last)/self.dt

        #Get Link State
        self.position_now, self.link_velocity = self._get_link_state()

        self.steps_left = np.copy(self.MAX_EPISODE)
        self.last_act = np.array([0,0,0])
        self.cur_done = False
        self.new_obs = self._get_obs(action = self.last_act, 
                                     desired_orientation_velocity = 0, 
                                     desired_linear_velocity = np.array([0,0,0]), 
                                     error = np.array([0,0,0]), 
                                     orientation_error = 0)
        return self.new_obs

    def render(self, mode='human'):
        print(f'State {self.new_obs}, action: {self.last_act}, done: {self.cur_done}')

    def _get_joint_state(self):
        theta0, theta1, theta2 = p.getJointStates(self.boxId, [2,3,4])
        theta_now = self.normalize(np.array([theta0[0], theta1[0], theta2[0]]))
        theta_dot_now = np.array([theta0[1], theta1[1], theta2[1]])
        return theta_now, theta_dot_now

    def normalize(self, x):
        return ((x+np.pi)%(2*np.pi)) - np.pi

    def _get_obs(self, action, desired_orientation_velocity, desired_linear_velocity, error, orientation_error):
        return np.concatenate(
            [
                self.theta_now,
                self.theta_dot_now,
                action,
                desired_linear_velocity,
                error,
                self.position_now,
                self.link_velocity,
                [self.orientation_now, self.orientation_velocity, desired_orientation_velocity, orientation_error]
            ]
        )

    def rot_mat(self, theta):
        return np.array([[np.cos(theta), 0, np.sin(theta)],
                         [0, 1, 0],
                         [-np.sin(theta), 0, np.cos(theta)]])
    
    def rotmat2theta(self, matrix):
        return np.arctan2(matrix[0,2],matrix[0,0])

    def homogeneous_transformation(self, theta):
        return np.array([[np.cos(theta), -np.sin(theta), 0, 0],
                         [np.sin(theta), np.cos(theta), 0, 0],
                         [0, 0, 1, 0],
                         [0, 0, 0, 1]])

    def _get_link_state(self):
        (linkWorldPosition,
        linkWorldOrientation,
        localInertialFramePosition,
        localInertialFrameOrientation,
        worldLinkFramePosition,
        worldLinkFrameOrientation,
        worldLinkLinearVelocity,
        worldLinkAngularVelocity) = p.getLinkState(self.boxId, 4, computeLinkVelocity=1, computeForwardKinematics=1)
        
        return linkWorldPosition, worldLinkLinearVelocity
    
    def _set_target(self, targets):
        position_target = targets[0]
        self.orientation_target = targets[1]
        self.steps_left = self.MAX_EPISODE

        theta_swing = np.arctan2(position_target[1],position_target[0])
        xyzw_position_target = np.concatenate((position_target,[1]))

        homogeneous_transformation_swing = self.homogeneous_transformation(theta_swing)

        xyzw_position_target_to_base = np.linalg.inv(homogeneous_transformation_swing)@xyzw_position_target
        self.position_target = xyzw_position_target_to_base[:-1]
        print(self.position_target)