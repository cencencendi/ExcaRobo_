import gym
import collections
import math
import numpy as np
import os
import time

from ExcaRobo_2 import ExcaRobo


SIM_ON = 0

if __name__ == "__main__":
    env = ExcaRobo(SIM_ON)

    episode = 1
    for i in range(1,episode+1):
        obs = env.reset()
        done = False
        score = 0

        while not done:
            env.render()
            action= env.action_space.sample()
            obs, reward, done, _ = env.step(action)
            score += reward
        print(f"Score: {score}")