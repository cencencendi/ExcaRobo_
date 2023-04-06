import gym
import math
import time
import numpy as np
import pybullet as p
from gym import spaces
import pybullet_data

class ExcaBot(gym.Env):
    def __init__(self, sim_active) :
        super(ExcaBot, self).__init__()
        self.sim_active = sim_active
        if self.sim_active:
               physicsClient = p.connect(p.GUI)#or p.DIRECT for non-graphical version
        else:
            physicsClient = p.connect(p.DIRECT)#or p.DIRECT for non-graphical version

        self.MAX_EPISODE = 1_000_000
        self.dt = 1.0/240.0
        # threshold = np.array([13,13,13,0], dtype = np.float32) #max x,y,z the end effector can reach
        self.min_obs = np.array([0,0,0,-3.1, -0.954, -0.1214, -0.32, -10.0, -10.0, -10.0, -10.0, -1000], dtype = np.float32)
        self.max_obs = np.array([13,13,13,3.1, 1.03, 1.51, 3.14, 10.0, 10.0, 10.0, 10.0, 0], dtype = np.float32)
        self.max_velocity = np.array([10.0, 10.0, 10.0, 10.0], dtype = np.float32)

        self.observation_space = spaces.Box(low =self.min_obs, high = self.max_obs, dtype=np.float32)
        self.action_space = spaces.Box(low = -self.max_velocity, high = self.max_velocity, dtype=np.float32)
        self.steps_left = self.MAX_EPISODE
        self.state = [0,0,0,0,0,0,0,0,0,0,0,0] #[x,y,z,theta0, theta1, theta2, theta3, thetadot0, thetadot1, thetadot2, thethadot3, norm_error] --> x,y,z end efector
        self.orientation = [0,0,0,0] #qarternion
        # self.theta_target = [1.5,-0.628,-0.125, 0.3] #theta0 = joint1, theta1 = joint2, theta2 = joint3, theta3 = joint4
        self.pose_target = np.array([8,0,4], dtype=np.float32)
        self.start_simulation()

    def step(self, action):
        action = np.clip(action, -self.max_velocity, self.max_velocity)
        p.setJointMotorControl2(self.boxId, 1 , p.VELOCITY_CONTROL, targetVelocity = action[0], force= 50_000)
        p.setJointMotorControl2(self.boxId, 2 , p.VELOCITY_CONTROL, targetVelocity = action[1], force= 250_000)
        p.setJointMotorControl2(self.boxId, 3 , p.VELOCITY_CONTROL, targetVelocity = action[2], force= 250_000)
        p.setJointMotorControl2(self.boxId, 4 , p.VELOCITY_CONTROL, targetVelocity = action[3], force= 250_000)

        #Update Simulations
        p.stepSimulation()
        time.sleep(self.dt)

        #Orientation (Coming Soon)

        #Calculate error
        self.theta_now = self._get_joint_state()
        self.pose_now = self._get_link_state(4)

        penalty = 0
        if np.any(self.theta_now > self.normalize(self.max_obs[3:7])) or np.any(self.theta_now < self.normalize(self.min_obs[3:7])):
            less_idx = np.argwhere(self.theta_now < self.normalize(self.min_obs[3:7]))[:,0]
            more_idx = np.argwhere(self.theta_now > self.normalize(self.max_obs[3:7]))[:,0]
            for less, more in zip(less_idx, more_idx):
                penalty += 0.1 * ((self.theta_now[less] - self.normalize(self.min_obs[3:7])[less])**2 + (self.theta_now[more] - self.normalize(self.max_obs[3:7])[more]**2))

        done = bool(self.steps_left<0)
        error = self.pose_now - self.pose_target
        norm_error = np.linalg.norm(error)**2
        if not done:
            self.reward = - (norm_error + 0.001*action[0]**2 + 0.001*action[1]**2 + 0.001*action[2]**2 + 0.001*action[3]**2 + penalty)
            self.steps_left -= 1

        #Update State
        self.state = np.concatenate((self.pose_now ,self.theta_now, np.array(action), np.array([norm_error])), axis=None)
        
        self.act = action
        self.cur_done = done
        return self.state,self.reward,done,{}
    def start_simulation(self):
        p.setAdditionalSearchPath(pybullet_data.getDataPath()) #optionally

        ## Setup Physics
        p.setGravity(0,0,-9.8)

        ## Load Plane
        planeId = p.loadURDF("plane.urdf")

        ## Load Robot
        startPos = [self.state[0],self.state[1],1.4054411813121799]
        startOrientation = p.getQuaternionFromEuler([0,0,0])
        self.boxId = p.loadURDF("aba_excavator/excavator.urdf",startPos, startOrientation)

    def reset(self):
        p.resetSimulation()
        self.start_simulation()
        self.state = [0,0,0,0,0,0,0,0,0,0,0,0]
        # self.theta_now = np.array([0.0,0.0,0.0,0.0], dtype = np.float32)
        self.theta_now = self._get_joint_state()
        self.pose_now = self._get_link_state(4)
        self.steps_left = self.MAX_EPISODE
        self.act = [0,0,0,0]
        self.cur_done = False
        return np.array([self.state])

    def render(self, mode='human'):
        print(f'State {self.state}, action: {self.act}, done: {self.cur_done}')

    def _get_joint_state(self):
        theta0, theta1, theta2, theta3 = p.getJointStates(self.boxId, [1,2,3,4])
        return self.normalize(np.array([theta0[0], theta1[0], theta2[0], theta3[0]]))

    def normalize(self, x):
        return ((x+np.pi)%(2*np.pi)) - np.pi
    
    def _get_link_state(self,link_numb):
        (linkWorldPosition,
            linkWorldOrientation,
            localInertialFramePosition,
            localInertialFrameOrientation,
            worldLinkFramePosition,
            worldLinkFrameOrientation,
            worldLinkLinearVelocity,
            worldLinkAngularVelocity) = p.getLinkState(self.boxId, link_numb, computeLinkVelocity=1, computeForwardKinematics=1)
        return np.array(linkWorldPosition, dtype=np.float32)
