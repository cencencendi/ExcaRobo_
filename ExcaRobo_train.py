import os
import numpy as np
from ExcaRobo_Final import ExcaRobo
from stable_baselines3 import PPO

SIM_ON = 0

if __name__ == "__main__":
    env = ExcaRobo(SIM_ON)

    log_path = os.path.join('Training', 'Logs', 'Tuned')
    model = PPO(policy='MlpPolicy', 
                env = env, 
                verbose=1, 
                tensorboard_log=log_path,
                batch_size= 1024)
    model.learn(total_timesteps=5e6, tb_log_name="bs:1024", log_interval=1)

    model_save_path = os.path.join('Training', 'Saved Models', 'Tuned', 'bs:1024')
    model.save(model_save_path)