##
# Derived from https://github.com/hill-a/stable-baselines/blob/master/README.md
#
##


from gym_miniworld.wrappers import *

import gym
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines.deepq.policies import MlpPolicy, CnnPolicy
from stable_baselines import DQN

from stable_baselines.bench import Monitor


#env = gym.make('MiniWorld-Maze-v0')
env = DummyVecEnv([lambda: Monitor(gym.make('MiniWorld-Maze-v0'),"./logs")])


model = DQN(CnnPolicy, env, verbose=1)
model.learn(total_timesteps=100000)

model.save("DQNCNNMAZE_EXPO2")

obs = env.reset()
for i in range(1000):
    action, _states = model.predict(obs)
    obs, rewards, dones, info = env.step(action)
    env.render()

env.close()