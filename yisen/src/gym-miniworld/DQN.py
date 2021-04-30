#!/usr/bin/env python3
"""
This script allows you to manually control the simulator
using the keyboard arrows.
"""
import time
import sys
import argparse
import pyglet
import math
from pyglet.window import key
from pyglet import clock
import numpy as np
import gym
import gym_miniworld
import cv2
import orbslam2
from stable_baselines.bench import Monitor
from stable_baselines.deepq.policies import MlpPolicy, CnnPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines import DQN



def step(action):
    global slam_frame_count
    print('step {}/{}: {}'.format(env.step_count+1, env.max_episode_steps, env.actions(action).name))

    obs, reward, done, info = env.step(action)

    slam_frame_count=slam_frame_count+1/10
    res=slam.process_image_mono(obs, slam_frame_count)
    print(res)
    if reward > 0:
        print('reward={:.2f}'.format(reward))

    if done:
        print('done!')
        env.reset()

    #env.render('pyglet', view=view_mode)

print(orbslam2.__file__)
parser = argparse.ArgumentParser()
parser.add_argument('--env-name', default='MiniWorld-Maze-v0')
parser.add_argument('--domain-rand', action='store_true', help='enable domain randomization')
parser.add_argument('--no-time-limit', action='store_true', help='ignore time step limits')
parser.add_argument('--top_view', action='store_true', help='show the top view instead of the agent view')
args = parser.parse_args()

vocab_path="/home/yisen/CS766/ORB_SLAM2/Vocabulary/ORBvoc.txt"
settings_path="/home/yisen/CS766/gym-miniworld/slam_settings.yaml"
slam = orbslam2.System(vocab_path, settings_path, orbslam2.Sensor.MONOCULAR)
slam.set_use_viewer(True)
slam.initialize()
slam_frame_count=0
if args.no_time_limit:
    env.max_episode_steps = math.inf
if args.domain_rand:
    env.domain_rand = True

view_mode = 'top' if args.top_view else 'agent'

env = gym.make(args.env_name)
#env = DummyVecEnv([lambda: Monitor(gym.make('MiniWorld-Maze-v0'),"./logs")])
# Optional: PPO2 requires a vectorized environment to run
# the env is now wrapped automatically when passing it to the constructor
# env = DummyVecEnv([lambda: env])

model = DQN(CnnPolicy, env, verbose=1)
model.learn(total_timesteps=100000)



obs = env.reset()
for i in range(1000):
    action, _states = model.predict(obs)
    obs, rewards, dones, info = env.step(action)
    env.render()

env.close()
