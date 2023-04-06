import os

from position_orientation2 import ExcaRobo
from stable_baselines3 import PPO

SIM_ON = 0

if __name__ == "__main__":
    env = ExcaRobo(SIM_ON)

    log_path = os.path.join('Training', 'Logs', 'Position_orientation')
    model = PPO('MlpPolicy', env, verbose=1, tensorboard_log=log_path)
    model.learn(total_timesteps=2_500_000, tb_log_name="rew: all_exp_obsspace: (11,)", log_interval=1)

    model_save_path = os.path.join('Training', 'Saved Models', 'Position_orientation', 'rew: all_exp_obsspace: (11,)')
    model.save(model_save_path)