from mss import mss # take screenshots
import pydirectinput # sinulate / mouse inputs
import cv2 as cv# picture editing
import numpy as np 
import pytesseract #OCR (Optical Character Recognition)
from matplotlib import pyplot as plt
import matplotlib.image as mpimg
from gymnasium import Env   #Magic behind RL
import gymnasium as gym
from gymnasium.spaces import Box, Discrete
import pyautogui
import time
from Trainer import TrainAndLoggingCallback
import tensorflow as tf


class WorldEnv(Env):
    def __init__(self):
        super().__init__()
        self.action_space = Discrete(3)
        self.cap = mss()
        self.observation_space = Box(low = 0, high = 255, shape=(1,100,120), dtype=np.uint8)
        self.gamelocation = {'top' : 600, 'left' : 0,'width' : 1200, 'height' : 700}
        self.doneLocation = {'top' : 550, 'left' : 800,'width' : 600, 'height' : 400}
    

    def reset(self , seed=None, options=None): 
        pydirectinput.leftClick(1422, 797)  
        info = {}
        return self.get_observation(), info

    def step(self, action):
        actions = {
            0 : "space",
            1 : "down",
            2 : "no_op"
        }
        if (action != 2):
            pydirectinput.press(actions[action])
        done, done_cap = self.getDone()  
        observation = self.get_observation()
        reward = 1 
        info = {}
        truncated = False
        return observation, reward, done, truncated, info
    

    def close(self):
        cv.destroyAllWindows()

    def get_observation(self):  
        pic = np.array(self.cap.grab(self.gamelocation))[:,:,:3].astype(np.uint8)
        gray = cv.cvtColor(pic, cv.COLOR_BGR2GRAY) 
        resized = cv.resize(gray, (120, 100))
        channel = np.reshape(resized, (1, 100, 120)) #channel has to be first for pytorch
        return channel


    def getDone(self):
        done_cap = np.array(self.cap.grab(self.doneLocation))
        doneString = ['GAME', 'OVER']
        done=False
        res = pytesseract.image_to_string(done_cap)[:4] # look for th first letter
        if res in doneString:
            done = True
        return done, done_cap


env = WorldEnv()

x, y = pyautogui.position()
print(x, y)

# plt.imshow(env.get_observation())
# plt.show()
   
episodes = 0

#Training loops. DO each episode till dino dies
for episode in range(1, episodes+1):
    obs = env.reset()
    done = False
    totalscore = 0  
 
    while not done: 

        obs, score, done, info = env.step(env.action_space.sample()) 
        totalscore += score
    print('Epsidoe:', episode, 'Reward:', totalscore )  
 
from stable_baselines3.common import env_checker
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack

# Check if Env is viable
env_checker.check_env(env)
env = WorldEnv()

logDir = './logs/'
checkpoint = './models/'       
callback = TrainAndLoggingCallback(check_freq=1000, save_path=checkpoint) 

from stable_baselines3 import DQN       
# model = DQN('CnnPolicy', env, tensorboard_log=logDir, verbose=1, buffer_size=1200000, learning_starts=1000)

# model.learn(total_timesteps=250000, callback=callback) 
m1 = DQN.load('./models/bestModel44000.zip')   

episodes = 10
for episode in range(1, episodes+1):
    obs = env.reset()
    done = False 
    obs= env.get_observation()  
   
    while not done:    
        action, _ = m1.predict(obs, deterministic=True) 
        action  = int(action)
        obs, _, done, _, _ = env.step(action)
    print('Loop vorbei' )                       