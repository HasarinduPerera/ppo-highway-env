import argparse
import numpy as np
import torch
from torch.distributions.categorical import Categorical
from environment import env
from ppo_utils import PPOUtils
from hyper import Hyperparameters


obs_space, action_space = Hyperparameters.all()[8:10]

# Define the command-line arguments
parser = argparse.ArgumentParser(description='Perform inference with PPO model')
parser.add_argument('-mp', type=str, required=True, help='Path to the pre-trained model')
parser.add_argument('-i', type=int, default=10, help='Number of inference iterations')

# Parse the command-line arguments
args = parser.parse_args()

# Load the model
actor_model, critic_model = PPOUtils.load_models(args.mp, obs_space, action_space)

rewards = []
avg_rewards = []
# Perform inference in the environment
for i in range(args.i): # Inference Iterations
    terminated = False
    truncated = False
    env.np_random
    observation, _ = env.reset()
    observation = observation.squeeze()
    # print(f'OBS: {observation}')
    while not terminated or truncated:
        with torch.no_grad():
            observation_tensor = torch.tensor(observation, dtype=torch.float32).reshape(1, -1)
            policy_logits = actor_model.policy(observation_tensor)  # Extract policy logits
            action_distribution = Categorical(logits=policy_logits)
            action = action_distribution.sample().item()

        observation, reward, terminated, truncated, _ = env.step(action)

        # Implement a way to detect Collisions.
        rewards.append(reward)
        env.render()

        ###
        # print(f'next_obs: {observation}')
        # print(f'reward: {reward}')
        # print(f'terminated: {terminated}')
        ###
    avg_reward = np.mean(rewards)
    avg_rewards.append(avg_reward)
    print(f'Episode: {i+1} Avg Reward: {avg_reward}')

print(f'Avg Reward for {i+1} Iteration(s): {np.mean(avg_rewards)}')

env.close()
