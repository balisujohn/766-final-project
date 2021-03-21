##
# Derived from https://github.com/hill-a/stable-baselines/blob/master/README.md
#
##


from gym_miniworld.wrappers import *

import gym
from stable_baselines.deepq.policies import MlpPolicy, CnnPolicy
from stable_baselines import DQN
from PIL import Image
import PIL
import numpy as np
import time


env = gym.make('MiniWorld-Hallway-v0', turn_step = 5)

#odel = DQN(CnnPolicy, env, verbose=1)
#model.learn(total_timesteps=100000)

model = DQN.load("DQNCNNHALLWAY")


result_string = "" 

count = 0
for e in range (1):
    obs = env.reset()
    for i in range(100):
        action, _states = model.predict(obs)
        obs, rewards, dones, info = env.step(action)
        raw_image = env.render(mode = "rgb_array")
        image_array = np.array(raw_image, dtype = np.uint8)
        image = Image.fromarray(image_array)
        image = image.resize((640,480), 1)
        time_string = '%.5f' % time.time()
        image.save(f'images/rgb/{time_string}.png')
        result_string = result_string + f"{time_string} rgb/{time_string}.png\n"
        print(obs)
        env.render()
        count += 1
manifest = open("./images/rgb.txt", "w")
manifest.write(result_string)
manifest.close()