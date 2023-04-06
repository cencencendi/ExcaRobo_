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

        self.MAX_EPISODE = 25_000
        self.dt = 1.0/240.0
        self.max_theta = [3.1, 1.03, 1.51, 3.14]    
        self.min_theta = [-3.1, -0.954, -0.1214, -0.32]
        self.max_angularVel = [0.5, 0.5, 0.5, 0.5]
        self.min_angularVel = [-0.5, -0.5, -0.5, -0.5]
        self.max_theta_error = [6.2, 1.984, 1.6314, 3.46]
        self.min_theta_error = [0,0,0,0]

        self.min_obs = np.array(   
                self.min_theta           +                       # Theta minimum each joint
                self.min_theta_error     +                       # Theta_error minimum each joint
                self.min_angularVel                              # theta_dot
        )

        self.max_obs = np.array(
                self.max_theta          +                       # Theta maximum each Joint
                self.max_theta_error    +                       # Theta_error maximum each joint
                self.max_angularVel                             # norm_error, penalty maximum (inf)
        )

        self.max_velocity = np.array(self.max_angularVel, dtype = np.float32)
        self.reward = 0
        self.observation_space = spaces.Box(low =self.min_obs, high = self.max_obs, dtype=np.float32)
        self.action_space = spaces.Box(low = -self.max_velocity, high = self.max_velocity, dtype=np.float32)
        self.steps_left = np.copy(self.MAX_EPISODE)
        self.state = [0,0,0,0] #[theta0, theta1, theta2, theta3]
        self.orientation = [0,0,0,0] #qarternion
        self.theta_target = [0,-0.4,-0.1, 1.1] #theta0 = joint1, theta1 = joint2, theta2 = joint3, theta3 = joint4
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

        self.new_obs = self._get_obs(self.theta_now, action)
        error = abs(self.new_obs[4:8])
        self.norm_error = np.mean(error)
        
        reward = (1-error[0])+(1-error[1])+(1-error[2])+(1-error[3])
        if self.norm_error<1e-2:
            reward = 1000

        if np.any(self.theta_now > np.array(self.max_theta)) or np.any(self.theta_now < np.array(self.min_theta)):
            done = True
            self.reward = -1000
        else:
            done = bool(self.steps_left<0)
            self.reward = reward - 0.1*(action[0]**2+action[1]**2+action[2]**2+action[3]**2)
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
        startPos = [self.state[0],self.state[1],1.4054411813121799]
        startOrientation = p.getQuaternionFromEuler([0,0,0])
        self.boxId = p.loadURDF("aba_excavator/excavator.urdf",startPos, startOrientation)

    def reset(self):
        p.resetSimulation()
        self.start_simulation()
        self.theta_now = self._get_joint_state()
        self.steps_left = self.MAX_EPISODE
        self.act = [0,0,0,0]
        self.cur_done = False
        self.new_obs = self._get_obs(self.theta_now, self.act)
        return self.new_obs

    def render(self, mode='human'):
        print(f'State {self.new_obs}, action: {self.act}, done: {self.cur_done}')

    def _get_joint_state(self):
        theta0, theta1, theta2, theta3 = p.getJointStates(self.boxId, [1,2,3,4])
        return self.normalize(np.array([theta0[0], theta1[0], theta2[0], theta3[0]]))

    def normalize(self, x):
        return ((x+np.pi)%(2*np.pi)) - np.pi

    def _get_obs(self, state, action):
        error_now = np.array(self.theta_target) - state
        return np.concatenate((state, error_now, np.array(action)), axis=None)
    


