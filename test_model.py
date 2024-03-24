from enviroment import WorldEnv
from stable_baselines3 import DQN

#Create Env to play the Game
env = WorldEnv()

#Load the Model to test
model_path = './models/bestModel44000.zip'
dqn_model = DQN.load(model_path)             

def model_tester(model, episodes):
    """Tests a trained Model.
    
    Args:
        model (): A trained ML Model.
        episodes (int): Number of episodes to train on.
    """

    for episode in range(1, episodes+1):
        obs = env.reset()
        done = False 
        obs= env.get_observation()  
    
        while not done:    
            action, _ = model.predict(obs, deterministic=True) 
            action  = int(action)
            obs, _, done, _, _ = env.step(action)
        print('Loop vorbei' )

#Test the model
model_tester(dqn_model, 10)                         