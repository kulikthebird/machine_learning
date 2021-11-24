import gym_gomoku
import gym
import ray
import ray.rllib.agents.dqn as dqn


def env_creator(env_name):
    import gym
    import gym_gomoku
    return gym.make('Gomoku9x9-v0')


ray.tune.register_env('Gomoku9x9-v0', env_creator)


CHECKPOINT_NUMBER = 1150
RAY_DIRECTORY = "/home/tomaszkulik/ray_results/DQN/"
CHECKPOINT_PATH = RAY_DIRECTORY + \
    "DQN_Gomoku9x9-v0_0537a_00002_2_lr=0.01_2021-04-04_14-55-23/checkpoint_{number}/checkpoint-{number}".format(
        number=CHECKPOINT_NUMBER)
config = {
    "env": 'Gomoku9x9-v0',
    "num_gpus": 0,
    "num_workers": 1,
    "lr": 0.01,
    "framework": "torch",
}
ray.init()
agent = dqn.DQNTrainer(config=config, env='Gomoku9x9-v0')
agent.restore(CHECKPOINT_PATH)


env = gym.make('Gomoku9x9-v0')


# play a game
obs = env.reset()
for _ in range(20):
    action = agent.compute_action(obs)
    observation, reward, done, info = env.step(action)
    env.render(mode=None)
    if done:
        print("Game is Over")
        print("Last Reward: ", reward)
        break
