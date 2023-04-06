from Excarobo_env import ExcaRobo
from stable_baselines3 import PPO
import numpy as np
import matplotlib.pyplot as plt
import pybullet as p
import time
from generator import get_trajectories

def homogeneous_transformation(theta):
    return np.array([[np.cos(theta), -np.sin(theta), 0, 0],
                     [np.sin(theta), np.cos(theta), 0, 0],
                     [0, 0, 1, 0],
                     [0, 0, 0, 1]])

def rot_mat(theta):
    return np.array([[np.cos(theta), -np.sin(theta), 0],
                     [-np.sin(theta), np.cos(theta), 0],
                     [0, 1, 0]])

def initial_position(env, model, obs):
    # Initial Position
    env._set_target((np.array([8.12,0,4.07]), -1.1))
    done = False
    while not done:
        action, _ = model.predict(obs)
        obs, _, done, _ = env.step(action)

    return obs

def _digging_operation(env, model, obs, target):
    x,y,z = target[0][0]
    theta_swing = np.arctan2(y,x)
    _move_swing(env, theta_swing)
    homogeneous_transformation_swing = homogeneous_transformation(theta_swing)
    ones = np.ones((len(target[0]),1))

    xyzw_world = np.concatenate((target[0],ones),axis=1)
    new_target = []

    for xyzw in xyzw_world:
        xyzw_ = np.linalg.inv(homogeneous_transformation_swing)@xyzw
        new_target.append(xyzw_[:-1])
    
    # Digging Operation
    position_targets, orientation_targets = np.array(new_target), target[1]
    for target in zip(position_targets,orientation_targets):
        env._set_target(target)
        done = False
        while not done:
            # env.render()
            action, _ = model.predict(obs)
            obs, reward, done, info = env.step(action)

    return obs

def _move_swing(env, swing_target):
    # Swing
    while True:
        theta = p.getJointState(env.boxId, 1)[0]
        vel = 0.5*(swing_target - theta)
        joint_control_swing(env, vel)
        p.stepSimulation()
        time.sleep(1.0/240.)
        if(swing_target - theta)<5e-2:
            joint_control_swing(env, 0)
            p.stepSimulation()
            time.sleep(1.0/240.)            
            break

def joint_control_swing(env, velocity):
    p.setJointMotorControl2(env.boxId, 1 , p.VELOCITY_CONTROL, targetVelocity = velocity, force=50_000)
    p.setJointMotorControl2(env.boxId, 2 , p.VELOCITY_CONTROL, targetVelocity = 0, force=250_000)
    p.setJointMotorControl2(env.boxId, 3 , p.VELOCITY_CONTROL, targetVelocity = 0, force=250_000)
    p.setJointMotorControl2(env.boxId, 4 , p.VELOCITY_CONTROL, targetVelocity = 0, force=250_000)

def release_material(env, model, obs, target):
    print("releasing")
    x,y,z = target
    theta_swing = np.arctan2(y,x)
    _move_swing(env, theta_swing)
    
    homogeneous_transformation_swing = homogeneous_transformation(theta_swing)
    xyzw_world = np.concatenate([target,[1]])
    xyzw_swing = np.linalg.inv(homogeneous_transformation_swing)@xyzw_world

    new_target = xyzw_swing[:-1]

    env._set_target((new_target, -1.39))
    done = False
    while not done:
        # env.render()
        action, _ = model.predict(obs)
        obs, reward, done, info = env.step(action)

    return obs


if __name__ == "__main__":
    env = ExcaRobo(render_mode='human')
    model = PPO.load('../Training/Saved Models/Tuned/6', env=env)
    obs = env.reset()

    trajectories, theta_swing = get_trajectories()
    release_target = np.array([9.19,4,1.71])

    obs = initial_position(env, model, obs)
    obs = _digging_operation(env, model, obs, trajectories)
    obs = release_material(env, model, obs, release_target)
    print(obs[12:15])
    p.disconnect()
