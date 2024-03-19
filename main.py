from mss import mss # take screenshots
import pydirectinput # sinulate / mouse inputs
import cv2 as cv# picture editing
import numpy as np 
import pytesseract #OCR (Optical Character Recognition)
from matplotlib import pyplot as plt
import matplotlib.image as mpimg
import time
from gymnasium import Env   #Magic behind RL
from gymnasium.spaces import Box, Discrete

class WorldEnv:
    def __init__(self):
        self.actions = ["up", "down"]
        self.episodeScore = 0
        self.done = False
        self._oberservationSpace = None
    

    def reset(self):
        self.episodeScore = 0
        pydirectinput.leftClick(500, 500)
        return self.y

    def step(self, action):
        pydirectinput.press(action)

        #Check for done
        doneString = ['GAME', 'OVER']
        img = cv.imread('/Pictre/bild.png')
        text = pytesseract.image_to_string(img)
        reward = 1


        return doneString, reward, action

    @property
    def getobservationSpace(self):
        
        with mss.mss() as sct:
            monitor = sct.monitors[1]  # Assuming you want to capture the first monitor
            screenshot_path = sct.shot(output='Environment.png')
            self._observationSpace = screenshot_path
        return self._observationSpace


    def take_action(self):
        pass



# test ScreenCapture
# with mss() as sct:
#     monitor = {""}
#     oberservationSpace = sct.shot(output='Enviroment')
# img = mpimg.imread(oberservationSpace)
# imgplot = plt.imshow(img)
# plt.show()



env = WorldEnv()

obs = env._oberservationSpace
img = mpimg.imread(obs)
imgplot = plt.imshow(img)
plt.show()
