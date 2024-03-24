import pyautogui

#usefulll comment to gain mouse position
x, y = pyautogui.position()

#Testing enviroment screenshot
# plt.imshow(env.get_observation())
# plt.show()

#test if enviroments work
# episodes = 0

#Training loops. DO each episode till dino dies
# for episode in range(1, episodes+1):
#     obs = env.reset()
#     done = False
#     totalscore = 0  
 
#     while not done: 

#         obs, score, done, info = env.step(env.action_space.sample()) 
#         totalscore += score
#     print('Epsidoe:', episode, 'Reward:', totalscore )  
#  
