##
# Derived from https://github.com/hill-a/stable-baselines/blob/master/README.md
#
##


from gym_miniworld.wrappers import *

import gym
from stable_baselines.deepq.policies import MlpPolicy, CnnPolicy
from stable_baselines import DQN


env = gym.make('MiniWorld-Hallway-v0')

#odel = DQN(CnnPolicy, env, verbose=1)
#model.learn(total_timesteps=100000)

model = DQN.load("DQNCNNHALLWAY")

while True:
    obs = env.reset()
    for i in range(100):
        action, _states = model.predict(obs)
        obs, rewards, dones, info = env.step(action)
        env.render()
