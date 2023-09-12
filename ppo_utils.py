import os
import datetime
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.distributions.categorical import Categorical
from environment import env
from model import ActorCriticNetwork
from hyper import Hyperparameters


action_type, obs_type = Hyperparameters.all()[10:12]
DEVICE = Hyperparameters.all()[0]

class PPOUtils():
    def rollout(model, env, max_steps=1000):
        """
        Performs a single rollout.
        Returns training data in the shape (n_steps, observation_shape)
        and the cumulative reward.
        """
        ### Create data storage
        train_data = [[], [], [], [], []] # obs, act, reward, values, act_log_probs
        #env.np_random
        seed = 10
        # seed = int(np.random.randint(100))
        obs = env.reset(seed=seed, options={})
        obs = env.reset(options={})
        obs, _ = obs
        obs = np.array(obs)
        # print(f'OBS: {obs}')
        # print(f'_: {_}')

        ep_reward = 0
        for _ in range(max_steps):
            obs = obs.reshape(1, -1)
            logits, val = model(torch.tensor([obs], dtype=torch.float32,
                                            device=DEVICE))
            act_distribution = Categorical(logits=logits)
            act = act_distribution.sample()
            act_log_prob = act_distribution.log_prob(act).item()

            act, val = act.item(), val.item()

            next_obs, reward, terminated, truncated, _ = env.step(act)
            

            ###
            # print(f'next_obs: {next_obs}')
            print(f'reward: {reward}')
            # print(f'terminated: {terminated}')
            ###

            env.render()

            for i, item in enumerate((obs, act, reward, val, act_log_prob)):
                train_data[i].append(item)

            obs = next_obs
            ep_reward += reward
            if terminated or truncated:
                break

        train_data = [np.asarray(x) for x in train_data]

        ### Do train data filtering
        train_data[3] = PPOUtils.calculate_gaes(train_data[2], train_data[3])

        return train_data, ep_reward

    def discount_rewards(rewards, gamma=0.99):
        """
        Return discounted rewards based on the given rewards and gamma param.
        """
        new_rewards = [float(rewards[-1])]
        for i in reversed(range(len(rewards)-1)):
            new_rewards.append(float(rewards[i]) + gamma * new_rewards[-1])
        return np.array(new_rewards[::-1])

    def calculate_gaes(rewards, values, gamma=0.99, decay=0.97):
        """
        Return the General Advantage Estimates from the given rewards and values.
        """
        next_values = np.concatenate([values[1:], [0]])
        deltas = [rew + gamma * next_val - val for rew, val, next_val in zip(rewards, values, next_values)]

        gaes = [deltas[-1]]
        for i in reversed(range(len(deltas)-1)):
            gaes.append(deltas[i] + decay * gamma * gaes[-1])

        return np.array(gaes[::-1])
    
    def create_directory():
        now = datetime.datetime.now()
        date_time = now.strftime("%Y-%m-%d_%H-%M-%S")
        act_n_obs = f'{action_type}_{obs_type}'
        env_folder = env.spec.id.replace("-", "_")  # Replace hyphens with underscores in the environment name
        folder_name = f"{date_time}_{env_folder}_{act_n_obs}"
        #os.makedirs(folder_name, exist_ok=True)
        return folder_name
    
    def save_models(model, ep_rewards, ppo):
        # Create a directory for saving models and plots
        model_folder = 'models'
        os.makedirs(model_folder, exist_ok=True)

        # Create a subdirectory for the current run
        output_folder = os.path.join(model_folder, PPOUtils.create_directory())
        os.makedirs(output_folder, exist_ok=True)
            
        # Plot episode rewards
        plt.figure(figsize=(10, 5))
        plt.plot(ep_rewards, label='Episode Rewards')
        plt.xlabel('Episode')
        plt.ylabel('Reward')
        plt.title('Training Progress')
        plt.legend()
        plt.grid(True)

        # Save the actor model
        actor_model_path = os.path.join(output_folder, 'actor_model.pth')
        torch.save(model.state_dict(), actor_model_path)

        # Save the critic model
        critic_model_path = os.path.join(output_folder, 'critic_model.pth')
        torch.save(ppo.ac.state_dict(), critic_model_path)

        # Save the plot as an image inside the directory
        plot_path = os.path.join(output_folder, 'training_progress.png')
        plt.savefig(plot_path)

        print("Saved actor model to:", actor_model_path)
        print("Saved critic model to:", critic_model_path)
        print("Saved training plot to:", plot_path)

        # Display the plot
        plt.show()

    def load_models(model_path, obs_space, action_space):
        # Load saved actor and critic models
        actor_model = ActorCriticNetwork(obs_space, action_space)  # Use the same architecture as during training
        critic_model = ActorCriticNetwork(obs_space, action_space)  # Use the same architecture as during training

        # Load the saved state dictionaries
        actor_model.load_state_dict(torch.load(model_path))
        critic_model.load_state_dict(torch.load(model_path))

        # Set the models to evaluation mode
        actor_model.eval()
        critic_model.eval()

        return actor_model, critic_model
