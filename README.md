# Proximal Policy Optimization (PPO) for Farama Foundation Highway Environment

![Python Version](https://img.shields.io/badge/Python-3.7%2B-blue.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

This repository contains an implementation of Proximal Policy Optimization (PPO) for the Farama Foundation Highway Environment. PPO is a reinforcement learning algorithm used to train agents to make sequential decisions, and it is applied to solve the specific task of navigating the Farama Foundation Highway Environment.

## Table of Contents
- [Introduction](#introduction)
- [Installation](#installation)
- [Usage](#usage)
- [Contributing](#contributing)
- [License](#license)

## Introduction

The Farama Foundation Highway Environment is a environment for training reinforcement learning agents. This repository provides a PPO implementation for training an agent to navigate this environment effectively. The implementation includes all the necessary components for training and evaluating the agent's performance.

## Installation

1. Clone this repository to your local machine:

   ```shell
   git clone https://github.com/HasarinduPerera/ppo-highway-env.git
   ```

2. Install the required dependencies:

   ```shell
   cd ppo-highway-env
   pip install -r requirements.txt
   ```

## Usage

1. Train the PPO agent:

   ```shell
   python main.py
   ```

   You can modify hyperparameters, training settings, and network architectures to suit your specific needs.

2. Evaluate the trained agent:

   ```shell
   python inference.py -mp /path/to/pre-trained-model -i 10
   ```

   The evaluation script will load the trained model and test the agent's performance in the environment.

    - -mp or --model-path: Path to the pre-trained model (required).
    - -i or --inference-iterations: Number of inference iterations (default: 10).

3. Adjust and fine-tune the code as needed for your specific use case.

## Contributing

Contributions are welcome! If you'd like to contribute to this project, please follow these steps:

1. Fork the repository
2. Create a new branch for your feature or bug fix
3. Make your changes and commit them with descriptive messages
4. Push your branch to your fork
5. Create a pull request to this repository's `main` branch

## License

This project is licensed under the GNU General Public License v3.0 - see the [LICENSE](LICENSE) file for details.