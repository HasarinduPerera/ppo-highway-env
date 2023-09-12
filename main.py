import numpy as np
import torch
from model import ActorCriticNetwork
from agent import PPOTrainer
from ppo_utils import PPOUtils
from environment import env
from hyper import Hyperparameters


# print(Hyperparameters.all())
DEVICE, n_episodes, print_freq, policy_lr, value_lr, target_kl_div, max_policy_train_iters, value_train_iters, obs_space, action_space = Hyperparameters.all()[:10]
# print(DEVICE, n_episodes, print_freq, policy_lr, value_lr, target_kl_div, max_policy_train_iters, value_train_iters)

model = ActorCriticNetwork(obs_space, action_space)
model = model.to(DEVICE)
# train_data, reward = rollout(model, env) # Test rollout function

# Init Trainer
ppo = PPOTrainer(
    model,
    policy_lr = policy_lr,
    value_lr = value_lr,
    target_kl_div = target_kl_div,
    max_policy_train_iters = max_policy_train_iters,
    value_train_iters = value_train_iters)

# Training loop
ep_rewards = []
for episode_idx in range(n_episodes):
  # Perform rollout
  train_data, reward = PPOUtils.rollout(model, env)
  ep_rewards.append(reward)

  # Shuffle
  permute_idxs = np.random.permutation(len(train_data[0]))

  # Policy data
  obs = torch.tensor(train_data[0][permute_idxs],
                     dtype=torch.float32, device=DEVICE)
  acts = torch.tensor(train_data[1][permute_idxs],
                      dtype=torch.int32, device=DEVICE)
  gaes = torch.tensor(train_data[3][permute_idxs],
                      dtype=torch.float32, device=DEVICE)
  act_log_probs = torch.tensor(train_data[4][permute_idxs],
                               dtype=torch.float32, device=DEVICE)

  # Value data
  returns = PPOUtils.discount_rewards(train_data[2])[permute_idxs]
  returns = torch.tensor(returns, dtype=torch.float32, device=DEVICE)

  # Train model
  ppo.train_policy(obs, acts, act_log_probs, gaes)
  ppo.train_value(obs, returns)

  if (episode_idx + 1) % print_freq == 0:
    print('=========================================')
    print('Episode {} | Avg Reward {:.1f}'.format(
        episode_idx + 1, np.mean(ep_rewards[-print_freq:])))
    print('=========================================')

# Save models and the reward plot
PPOUtils.save_models(model, ep_rewards, ppo)

# env.close()
# show_videos()