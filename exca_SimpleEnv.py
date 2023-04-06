import gym
import math
import time
import numpy as np
import pybullet as p
from gym import spaces
import pybullet_data

class ExcaBot(gym.Env):
    def __init__(self, sim_active):
        super(ExcaBot, self).__init__()
        self.sim_active = sim_active
        if self.sim_active:
               physicsClient = p.connect(p.GUI)#or p.DIRECT for non-graphical version
        else:
            physicsClient = p.connect(p.DIRECT)#or p.DIRECT for non-graphical version

        self.MAX_EPISODE = 2000
        self.dt = 1.0/240.0
        self.max_theta = 3.1
        self.min_theta = -3.1
        self.max_angularVel = 0.5
        self.min_angularVel = -0.5
        self.max_theta_error = 2*np.pi
        self.min_theta_error = -2*np.pi

        self.min_obs = np.array([self.min_theta, self.min_theta_error, self.min_angularVel])

        self.max_obs = np.array([self.max_theta, self.max_theta_error, self.max_angularVel])

        self.max_velocity = np.array(self.max_angularVel, dtype = np.float32)
        self.reward = 0
        self.observation_space = spaces.Box(low =self.min_obs, high = self.max_obs, dtype=np.float32)
        self.action_space = spaces.Box(low = -self.max_velocity, high = self.max_velocity, dtype=np.float32)
        self.steps_left = np.copy(self.MAX_EPISODE)
        self.state = 0 #[theta0, theta1, theta2, theta3]
        self.orientation = [0,0,0,0] #quarternion
        self.theta_target = -0.4 #theta_1
        self.start_simulation()

    def step(self, action):
        action = np.clip(action, -self.max_velocity, self.max_velocity)
        # p.setJointMotorControl2(self.boxId, 1 , p.VELOCITY_CONTROL, targetVelocity = action, force= 50_000)
        p.setJointMotorControl2(self.boxId, 2 , p.VELOCITY_CONTROL, targetVelocity = action, force= 250_000)
        # p.setJointMotorControl2(self.boxId, 3 , p.VELOCITY_CONTROL, targetVelocity = action, force= 250_000)
        # p.setJointMotorControl2(self.boxId, 4 , p.VELOCITY_CONTROL, targetVelocity = action, force= 250_000)

        #Update Simulations
        p.stepSimulation()
        time.sleep(self.dt)

        #Orientation (Coming Soon)

        #Calculate error
        self.theta_now = self._get_joint_state()

        self.new_obs = self._get_obs(self.theta_now, action)
        error = self.new_obs[1]**2
        
        reward = -error

        if (self.theta_now > self.max_theta) or (self.theta_now < self.min_theta):
            self.reward = -1000
            done = True
        else:
            done = bool(self.steps_left<0)
            self.reward = reward
            self.steps_left -= 1

        #Update State
        
        self.act = action
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
        self.boxId = p.loadURDF("aba_excavator/excavator.urdf",startPos, startOrientation)
        # for i in range(500):
        #     p.setJointMotorControl2(self.boxId, 2 , p.POSITION_CONTROL, targetPosition = -0.45, force= 250_000)
        #     p.stepSimulation()

    def reset(self):
        p.resetSimulation()
        self.start_simulation()
        self.theta_now = self._get_joint_state()
        self.steps_left = self.MAX_EPISODE
        self.act = 0
        self.cur_done = False
        self.new_obs = self._get_obs(self.theta_now, self.act)
        return self.new_obs

    def render(self, mode='human'):
        print(f'State {self.new_obs}, action: {self.act}, done: {self.cur_done}')

    def _get_joint_state(self):
        theta = p.getJointState(self.boxId, 2)
        return self.normalize(theta[0])

    def normalize(self, x):
        return ((x+np.pi)%(2*np.pi)) - np.pi

    def _get_obs(self, state, action):
        error_now = self.rotmat2theta(self.rot_mat(self.theta_target)@self.rot_mat(state).T)
        return np.array([state, error_now, action])

    def rot_mat(self, theta):
        return np.array([[np.cos(theta), 0, np.sin(theta)],
                         [0, 1, 0],
                         [-np.sin(theta), 0, np.cos(theta)]])
    
    def rotmat2theta(self, matrix):
        return np.arctan2(matrix[0,2],matrix[0,0])
    


