import os

from onlyposition2 import ExcaRobo
from stable_baselines3 import PPO

SIM_ON = 0

if __name__ == "__main__":
    env = ExcaRobo(SIM_ON)

    log_path = os.path.join('Training', 'Logs', 'Position_only')
    model = PPO('MlpPolicy', env, verbose=1, tensorboard_log=log_path)
    model.learn(total_timesteps=2_500_000, tb_log_name="rw: exp_obsspace: only_error", log_interval=1)

    model_save_path = os.path.join('Training', 'Saved Models', 'Position_only_obsspace: only_error')
    model.save(model_save_path)