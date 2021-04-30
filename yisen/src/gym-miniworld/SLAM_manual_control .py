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
print(orbslam2.__file__)
parser = argparse.ArgumentParser()
parser.add_argument('--env-name', default='MiniWorld-Maze-v0')
parser.add_argument('--domain-rand', action='store_true', help='enable domain randomization')
parser.add_argument('--no-time-limit', action='store_true', help='ignore time step limits')
parser.add_argument('--top_view', action='store_true', help='show the top view instead of the agent view')
args = parser.parse_args()
total_time=0
total_step=0
temp_time=0
frame_per_action=10
env = gym.make(args.env_name)


print("2333333",total_time)
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

env.reset()

# Create the display window
env.render('pyglet', view=view_mode)

def step(action):
    global slam_frame_count 
    print('step {}/{}: {}'.format(env.step_count+1, env.max_episode_steps, env.actions(action).name))

    obs, reward, done, info = env.step(action)
    full_image=env.render('pyglet',view='agent')
    slam_frame_count=slam_frame_count+1/frame_per_action
    res=slam.process_image_mono(full_image, slam_frame_count)
    print(res)
    if reward > 0:
        print('reward={:.2f}'.format(reward))

    if done:
        print('done!')
        env.reset()

    env.render('pyglet', view=view_mode)

@env.unwrapped.window.event
def on_key_press(symbol, modifiers):
    global total_time
    global total_step
    """
    This handler processes keyboard commands that
    control the simulation
    """

    if symbol == key.BACKSPACE or symbol == key.SLASH:
        print('RESET')
        env.reset()
        env.render('pyglet', view=view_mode)
        return

    if symbol == key.ESCAPE:
        env.close()
        sys.exit(0)

    if symbol == key.UP:
        temp_time=time.time()
        for idx in range(frame_per_action):
            step(env.actions.move_forward)
        temp_time=time.time()-temp_time
        total_time=total_time+temp_time
        total_step=total_step+frame_per_action
        print("one action: ",temp_time)

    elif symbol == key.DOWN:
        temp_time=time.time()
        for idx in range(frame_per_action):
            step(env.actions.move_back)
        temp_time=time.time()-temp_time
        total_time=total_time+temp_time
        total_step=total_step+frame_per_action
        print("one action: ",temp_time)


    elif symbol == key.LEFT:
        temp_time=time.time()
        for idx in range(frame_per_action):
            step(env.actions.turn_left)
        temp_time=time.time()-temp_time
        total_time=total_time+temp_time
        total_step=total_step+frame_per_action
        print("one action: ",temp_time)

    elif symbol == key.RIGHT:
        temp_time=time.time()
        for idx in range(frame_per_action):
            step(env.actions.turn_right)
        temp_time=time.time()-temp_time
        print("one action: ",temp_time)
        total_time=total_time+temp_time
        total_step=total_step+frame_per_action


    elif symbol == key.PAGEUP or symbol == key.P:
        step(env.actions.pickup)
    elif symbol == key.PAGEDOWN or symbol == key.D:
        step(env.actions.drop)

    elif symbol == key.ENTER:
        step(env.actions.done)

@env.unwrapped.window.event
def on_key_release(symbol, modifiers):
    pass

@env.unwrapped.window.event
def on_draw():
    #env.render('pyglet', view=view_mode)
    pass

@env.unwrapped.window.event
def on_close():
    pyglet.app.exit()

total_time=0
total_step=0
temp_time=0
# Enter main event loop
pyglet.app.run()

env.close()
