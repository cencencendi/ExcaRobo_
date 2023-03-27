# from ExcaRobo_4 import ExcaRobo
# from ExcaRobo_3 import ExcaRobo
# from ExcaRobo_2 import ExcaRobo
# from ExcaRobo_1 import ExcaRobo
from ExcaRobo_Final import ExcaRobo
# from ExcaRobo_Final2 import ExcaRobo

from stable_baselines3 import PPO
import pybullet as p
import numpy as np
import matplotlib.pyplot as plt

SIM_ON = 1

if __name__ == "__main__":
    env = ExcaRobo(SIM_ON)
    model = PPO.load('Training/Saved Models/Tuned/1', env=env)
    obs = env.reset()
    score = 0
    done = False
    step = 0
    error_pose = []
    error_ori = []
    while not done:
        env.render()
        action, _ = model.predict(obs)
        obs, reward, done, info = env.step(action)
        score += reward
        step+=1
        error_pose.append(obs[12:15])
        error_ori.append(obs[24])
    p.disconnect()
    
    plt.figure(1, figsize=(4,2.5))
    plt.plot(list(range(len(error_ori))),error_ori)
    plt.title("Orientation error")
    
    plt.figure(2, figsize=(4,2.5))
    for i in range(3):
        plt.plot(list(range(len(error_ori))), np.array(error_pose)[:,i], label="x" if i==0 else "y" if i==1 else "z")
    plt.legend()
    plt.title("Position error")
    plt.show()

    print(f"Score: {score}, with step: {step}, pose: {obs[15:18]}, orientation: {obs[21]}\n")
    print(f"Position target: {env.position_target}, Orientation target: {env.orientation_target}\n\
            idx: {env.idx_target}")
    print(f"Position Error: {env.new_obs[12:15]}, \nOrientation  Error: {env.new_obs[24]}")
    low = np.argmin(abs(np.array(error_ori)))
    print(f"Lowest Error at index: {low}\
            which is: {error_ori[low]}\
            and position: {error_pose[low]}")


"""Best model 6_PPO6, action space [-0.3,0.3] shape=(3,)"""