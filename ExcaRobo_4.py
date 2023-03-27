import gym
import time
import numpy as np
import pybullet as p
from gym import spaces
import pybullet_data

class ExcaRobo(gym.Env):
    def __init__(self, sim_active):
        super(ExcaRobo, self).__init__()
        self.sim_active = sim_active
        if self.sim_active:
            physicsClient = p.connect(p.GUI)#or p.DIRECT for non-graphical version
        else:
            physicsClient = p.connect(p.DIRECT)#or p.DIRECT for non-graphical version

        self.MAX_EPISODE = 5_000
        self.dt = 1.0/240.0
        self.max_theta = [1.03, 1.51, 3.14]    
        self.min_theta = [-0.954, -0.1214, -0.32]
        self.position_target = np.array([9.817,0,0.942]) #theta0 = joint1, theta1 = joint2, theta2 = joint3, theta3 = joint4
        self.orientation_target =  -1.841
        # self.max_obs = np.concatenate(
        #     [
        #         np.array([1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0]),
        #         np.array([np.inf, np.inf, np.inf, np.inf]),
        #         np.array([0.1,0.1,0.1]),
        #         np.array([np.inf, np.inf, np.inf, np.inf])
        #     ]
        # )
        # self.min_obs = np.concatenate(
        #     [
        #         np.array([-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0]),
        #         np.array([np.inf, np.inf, np.inf, np.inf]),
        #         np.array([-0.1,-0.1,-0.1]),
        #         np.array([-np.inf, -np.inf, -np.inf, np.inf])
        #     ]
        # )
        self.observation_space = spaces.Box(low =-np.inf, high = np.inf, shape= (25,), dtype=np.float32)
        self.action_space = spaces.Box(low = -0.1, high = 0.1, shape=(3,), dtype=np.float32)
        self.steps_left = np.copy(self.MAX_EPISODE)
        self.state = np.zeros(5) #[theta1, theta2, x, y, z]
        
        self.start_simulation()

    def step(self, action):
        # p.setJointMotorControl2(self.boxId, 1 , p.VELOCITY_CONTROL, targetVelocity = action[0], force= 50_000)
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
        self.position_now, self.link_velocity = self._get_link_state()

        vec = np.array(self.position_now) - self.position_target
        desired_linear_velocity = -5*vec

        reward_dist = 4*np.exp(-np.linalg.norm(desired_linear_velocity-self.link_velocity))
        reward_orientation = -0.02*(desired_orientation_velocity-self.orientation_velocity)**2
        reward_ctrl = -0.0075*np.linalg.norm(action)

        reward = reward_dist + reward_ctrl + reward_orientation + 0.025
        self.new_obs = self._get_obs(action, desired_orientation_velocity, desired_linear_velocity, vec, orientation_error)

        if np.any(self.theta_now > np.array(self.max_theta)) or np.any(self.theta_now < np.array(self.min_theta)):
            done = True
            punishment = -1000
            self.reward = reward+punishment
        else:
            done = bool(self.steps_left<0)
            self.steps_left -= 1
            self.reward = reward
        #Update State
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
        startPos = [self.state[0],self.state[1],1.4054411813121799]
        startOrientation = p.getQuaternionFromEuler([0,0,0])
        self.boxId = p.loadURDF("aba_excavator/excavator.urdf",startPos, startOrientation)

    def reset(self):
        p.resetSimulation()
        self.start_simulation()
        self.theta_now, self.theta_dot_now = self._get_joint_state()
        self.orientation_last = self.theta_now[-1]
        self.steps_left = np.copy(self.MAX_EPISODE)
        self.last_act = [0,0,0]
        self.cur_done = False
        self.new_obs = np.zeros(25)
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