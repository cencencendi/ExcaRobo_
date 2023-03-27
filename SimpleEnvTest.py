import pybullet as p
import numpy as np
import matplotlib.pyplot as plt
# from onlyposition import ExcaRobo
from onlyposition2 import ExcaRobo
# from onlyposition3 import ExcaRobo
from stable_baselines3 import PPO

SIM_ON = 1

def where_steady(array):
    idx_steady = None
    counter = 0
    for idx, numb in enumerate(array):
        if np.all(abs(numb) < 0.15):
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
    model = PPO.load('Training/Saved Models/Position_only_obsspace: only_error', env=env)
    obs = env.reset()
    score = 0
    done = False
    step = 0
    position_error = []
    while not done:
        # env.render()
        action, _ = model.predict(obs)
        obs, reward, done, info = env.step(action)
        score += reward
        step+=1
        position_error.append(obs[-3:])
    # p.disconnect()

    position_error = np.array(position_error)
    idx_steady = where_steady(position_error)
    for i in range(3):
        plt.plot(list(range(step)), position_error[:,i], label = "x" if i==0 else "y" if i==1 else "z")
    plt.legend()
    plt.xlabel("Episode")
    plt.ylabel("Error")
    plt.title("Obs_Space: (3,) , Reward = dist+ctrl")
    plt.show()
    print(f"steady at episode: {idx_steady}")
    print(f"Score: {score}, with step: {step}, steady_state MSE: {(position_error[idx_steady:,:]**2).mean(axis=0)}")
    