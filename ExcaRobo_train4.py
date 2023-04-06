import os

from ExcaRobo_4 import ExcaRobo
from stable_baselines3 import PPO

SIM_ON = 0

if __name__ == "__main__":
    env = ExcaRobo(SIM_ON)

    log_path = os.path.join('Training', 'Logs', 'All_Pose')
    model = PPO('MlpPolicy', env, verbose=1, tensorboard_log=log_path)
    model.learn(total_timesteps=6_000_000)

    model_save_path = os.path.join('Training', 'Saved Models', 'All_Pose', '4_(4)_PPO30')
    model.save(model_save_path)
    print("Kelar brou")