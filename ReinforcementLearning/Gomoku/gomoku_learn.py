import ray
from ray.tune.logger import DEFAULT_LOGGERS


def env_creator(env_name):
    import gym
    import gym_gomoku
    return gym.make('Gomoku9x9-v0')

ray.tune.register_env('Gomoku9x9-v0', env_creator)


# TODO: Add custom model with sparse output (9x9).
# TODO: Add auxilary rewards


ray.tune.run("DQN",
             stop={
                 "episode_reward_mean": 80
             },
             loggers=DEFAULT_LOGGERS,
             config={
                 "env": 'Gomoku9x9-v0',
                 "num_gpus": 0,
                 "num_workers": 1,
                 "lr": ray.tune.grid_search([0.005, 0.02, 0.01]),
                 "framework": "torch",
             },
             checkpoint_freq=10,
             checkpoint_at_end=True,

            #  resume=True
             )



class KP0ActionMaskModel(TFModelV2):
     
    def __init__(self, obs_space, action_space, num_outputs,
        model_config, name, true_obs_shape=(11,),
        action_embed_size=5, *args, **kwargs):
         
        super(KP0ActionMaskModel, self).__init__(obs_space,
            action_space, num_outputs, model_config, name, 
            *args, **kwargs)
         
        self.action_embed_model = FullyConnectedNetwork(
            spaces.Box(0, 1, shape=true_obs_shape), 
                action_space, action_embed_size,
            model_config, name + "_action_embedding")
        self.register_variables(self.action_embed_model.variables())
 
    def forward(self, input_dict, state, seq_lens):
        avail_actions = input_dict["obs"]["avail_actions"]
        action_mask = input_dict["obs"]["action_mask"]
        action_embedding, _ = self.action_embed_model({
            "obs": input_dict["obs"]["state"]})
        intent_vector = tf.expand_dims(action_embedding, 1)
        action_logits = tf.reduce_sum(avail_actions * intent_vector,
            axis=1)
        inf_mask = tf.maximum(tf.log(action_mask), tf.float32.min)
        return action_logits + inf_mask, state
 
    def value_function(self):
