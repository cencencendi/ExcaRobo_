import os

from ExcaRobo_4 import ExcaRobo
from stable_baselines3 import PPO

SIM_ON = 0

if __name__ == "__main__":
    env = ExcaRobo(SIM_ON)

    log_path = os.path.join('Training', 'Logs', 'End')
    model = PPO(policy='MlpPolicy', 
                env = env, 
                verbose=1, 
                tensorboard_log=log_path,
                batch_size= 1024,
                learning_rate=1e-4)
    model.learn(total_timesteps=1e7, tb_log_name="bs:1024_lr:1e-4_all_exp_unclipped", log_interval=1)

    model_save_path = os.path.join('Training', 'Saved Models', 'End', 'bs:1024_lr:1e-4_all_exp_unclipped')
    model.save(model_save_path)