##
# Derived from https://github.com/hill-a/stable-baselines/blob/master/README.md
#
##


from gym_miniworld.wrappers import *

import gym
from stable_baselines.deepq.policies import MlpPolicy, CnnPolicy
from stable_baselines import DQN


env = gym.make('MiniWorld-Maze-v0')

model = DQN(CnnPolicy, env, verbose=1)
model.learn(total_timesteps=1000000)

model.save("DQNCNNMAZE")

obs = env.reset()
for i in range(1000):
    action, _states = model.predict(obs)
    obs, rewards, dones, info = env.step(action)
    env.render()

env.close()