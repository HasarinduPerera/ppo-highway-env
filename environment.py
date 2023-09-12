import pprint
import gymnasium as gym
from hyper import Hyperparameters


action_type, obs_type = Hyperparameters.all()[10:12]

env = gym.make('highway-v0', render_mode='rgb_array')
# pprint.pprint(env.config)

'''
env.configure({
    'action': {'type': 'ContinuousAction'},
    'observation': {'type': 'GrayscaleObservation',
                    #'features': ['y'],
                    "observation_shape": (64, 64),
                    "stack_size": 4,
                    "weights": [0.2989, 0.5870, 0.1140],  # weights for RGB conversion
                    "scaling": 1.75,
                },
    'policy_frequency': 2
    ,
})
'''
env.configure({
    'action': {'type': action_type},
    'observation': {'type': obs_type,
                    'features': ["presence", "x", "y", "vx", "vy", "cos_h", "sin_h"],
                    #'features': ["x", "y", "vx", "vy", "cos_h", "sin_h"],
                    #'scales': [100, 100, 5, 5, 1, 1],
                    #'normalize': False,
                },
    'policy_frequency': 10,
    'show_trajectories': True,
    'lanes_count': 6,
    'vehicles_density': 2,
    'vehicles_count': 50,
    'screen_height': 300,
    'screen_width': 700,
})

# pprint.pprint(env.config)

# env = record_videos(env)
obs_space_size = env.observation_space
action_space_size = env.action_space
print(f'Action Space: {env.action_space}')
print(f'Observation Space: {env.observation_space}')
# print(f'Available Actions: {env.get_available_actions()}')

