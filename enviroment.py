from mss import mss # take screenshots
import pydirectinput # sinulate / mouse inputs
import cv2 as cv# picture editing
import numpy as np 
import pytesseract #OCR (Optical Character Recognition)
from gymnasium import Env   #Magic behind RL
from gymnasium.spaces import Box, Discrete
import pyautogui



class WorldEnv(Env):
    def __init__(self):
        super().__init__()
        self.action_space = Discrete(3)
        self.cap = mss()
        self.observation_space = Box(low = 0, high = 255, shape=(1,100,120), dtype=np.uint8)
        self.gamelocation = {'top' : 600, 'left' : 0,'width' : 1200, 'height' : 700}
        self.doneLocation = {'top' : 550, 'left' : 800,'width' : 600, 'height' : 400}
    

    def reset(self , seed=None, options=None): 
        """Resets the Game.

        Return:
            get_observation() (NDArray): Contains an image of the enviroment.
            info (dict): Empty dict used, since Gym expects an inf output.
        """
        pydirectinput.leftClick(1422, 797)  
        info = {}
        return self.get_observation(), info

    def step(self, action):
        """Performs an action in the Game.
        
        Args:
            actions (int): Giving action to do.

        Return:
            observation (NDArray): Picture of the current game progress.
            reward (int): Reward for current episode.
            done (bool): Status if game is over.
            truncated (False): Wont ever be truncated.
            info: Empty value, since gymnasium expects 5 outputs.
            """
        actions = {
            0 : "space",
            1 : "down",
            2 : "no_op"
        }
        if (action != 2):
            pydirectinput.press(actions[action])
        done = self.getDone()  
        observation = self.get_observation()
        reward = 1 
        info = {}
        truncated = False
        return observation, reward, done, truncated, info
    

    def close(self):
        cv.destroyAllWindows()

    def get_observation(self):  
        """Get an Screenshot of the game.
        
        Return:
            NDArray: Screenshot as array.
        """
        pic = np.array(self.cap.grab(self.gamelocation))[:,:,:3].astype(np.uint8)
        gray = cv.cvtColor(pic, cv.COLOR_BGR2GRAY) 
        resized = cv.resize(gray, (120, 100))
        channel = np.reshape(resized, (1, 100, 120)) #channel has to be first for pytorch
        return channel


    def getDone(self): 
        """Figure out if game over is reached.
        
        Return: 
            bool: Status wether game is over or not.
        """
        done_cap = np.array(self.cap.grab(self.doneLocation))
        doneString = ['GAME', 'OVER']
        done=False
        res = pytesseract.image_to_string(done_cap)[:4] # look for th first letter
        if res in doneString:
            done = True
        return done

