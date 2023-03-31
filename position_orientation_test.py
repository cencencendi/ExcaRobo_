import pybullet as p
import numpy as np
import matplotlib.pyplot as plt
# from position_orientation2 import ExcaRobo
# from position_orientation2 import ExcaRobo
from position_orientation_vel import ExcaRobo
from stable_baselines3 import PPO

SIM_ON = 0

def where_steady(array):
    idx_steady = None
    counter = 0
    for idx, numb in enumerate(array):
        if np.all(abs(numb) < 0.2):
            idx_steady = idx_steady or idx
            counter += 1
        else:
            idx_steady = None
            counter = 0

        if counter >= len(array)/5:
            return idx_steady
    return None        

if __name__ == "__main__":
    env = ExcaRobo(SIM_ON)
    model = PPO.load('Training/Saved Models/Position_orientation/rew: velocity_tracking obsspace: (22,)', env=env)
    obs = env.reset()
    score = 0
    done = False
    step = 0
    position_error = []
    rew = []
    orientation_error = []
    while not done:
        # env.render()
        action, _ = model.predict(obs)
        obs, reward, done, info = env.step(action)
        score += reward
        step+=1
        rew.append(reward)
        position_error.append(obs[9:12])
        orientation_error.append(obs[-1])
    p.disconnect()

    position_error = np.array(position_error)
    idx_steady = where_steady(position_error)
    plt.figure(1)
    for i in range(3):
        plt.plot(list(range(step)), position_error[:,i], label = "x" if i==0 else "y" if i==1 else "z")
    plt.legend()
    plt.xlabel("Episode")
    plt.ylabel("Error")
    plt.title("Obs_Space: (3,) , Reward = dist+ctrl")

    plt.figure(2)
    plt.plot(list(range(step)), rew)

    plt.figure(3)
    plt.plot(list(range(step)), orientation_error)

    plt.show()

    print(f"steady at episode: {idx_steady}")
    print(f"Score: {score}, with step: {step}, steady_state MSE: {np.sqrt((position_error[idx_steady:,:]**2).mean(axis=0))}")
    