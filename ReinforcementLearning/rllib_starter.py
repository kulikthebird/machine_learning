# import random
# import gym 
# env = gym.make("CartPole-v1")
# done = False
# env.reset() 
# while not done:
#     env.step(random.choice(range(env.action_space.n)))
#     env.render()



import ray.tune as tune
from ray.tune.logger import DEFAULT_LOGGERS


# get_location
# self.action_space = [] # 2
# self.observation_space
# class MyEnv:
# don't use preprocessor
# define step function
# self.action_space = Discrete(2) 
# _, reward, done, _ = self.env.step(action) return self.get_screen(), reward, done, {} 
# def step(self, action): _, reward, done, _ = self.env.step(action) return self.get_screen(), reward, done, {}
# self.action_space = Discrete(2) 
# _, reward, done, _ = self.env.step(action) return self.get_screen(), reward, done, {} 
# def step(self, action): _, reward, done, _ = self.env.step(action) return self.get_screen(), reward, done, {} 
# self.observation_space 
# gymspaces box
# gym.spaces.Box(0, 300, shape=(1, xxx)) 
# # def reset():

tune.run("DQN", 
         stop= {
             "episode_reward_mean": 200
         },
         loggers = DEFAULT_LOGGERS,
         config= { 
             "env": "CartPole-v0",
             "num_gpus": 0,
             "num_workers": 1,
             "lr": tune.grid_search([0.01, 0.001, 0.0001]),
            #  "eager": False,
             "framework": "torch",
            #  
         },
        )
