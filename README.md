# Manage a Farm using Reinforcement learning

The repository contains the project for the course of reinforcement learning at Unimi - Data Science for Economics (A.Y. 23/24)

## Installation
The repository contains a `requirements.txt` file which can be used to install the packages and libraries used to run the code

## Usage & code structure
The repository contains the Python scripts with the environment definition, different names correspond to different versions of the environment implementation all compatible with the agents defined.
Two agents are implemented in two separate Python files, a REINFORCE agent in `REINFORCE_agent.py` and a Deep Q-Learning agent in `DQN_agent.py`. The deep Q-learning agent can run with different training modes that are explained directly in the script. The agents are trained running the `TrainingDQN.py` and `training_REINFORCE.py`  

The `optimization.py` contains the code to run hyperparameters optimization with `optuna`, which can optimize both the networks hyperparameters and the RL proper parameters.
The `training.py` file contains functions and methods that are needed to run the hyperoptimization algorithm.

The notebooks contains the data analysis on the trained agents and on their training process.

## Problem
You are the manager of a farm. You have an initial budget of 2000 €. Each year you have to take some decisions about how to invest you money, but you can do only one of the following things:
1. Buy one sheep: a sheep costs 1000 €
2. Growing wheat: when you choose this action, you spend 20 €

At the end of the year, you harvest the wheat and you sell your wool. Each sheep produces 1 wool unit that is sold for 10 €. Moreover the number of sheeps evolve following abinomial distribution given by $$N_t = N_0 (1.15)^t $$.
Selling the harvested wheat instead gives you 50 €. However, during the year, there is a probability $\alpha = 30%$ that your fields are devastated by a storm. In this case, your harvest will give you 0 €.

Your manager career ends if you run out of money or, in any case, after 30 years, when you will retire.

You want to have a long and prosperous career.
