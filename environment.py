import numpy as np
import gymnasium as gym
from gymnasium import spaces
from gymnasium.envs.registration import register

class TabularFarmEnv(gym.Env):

    def __init__(self, initial_sheep=0, initial_budget=2000, initial_year=0,
                 prob_storm=0.3, prob_sheep=0.1, sheep_cost=1000, wheat_cost=20):
        
        super().__init__()

        self.initial_budget = initial_budget
        self.initial_year = initial_year
        self.initial_sheep = initial_sheep
        self.prob_storm = prob_storm
        self.prob_sheep = prob_sheep
        self.sheep_cost = sheep_cost
        self.wheat_cost = wheat_cost

        self.max_year = 30
        self.max_sheep = 70  # Maximum number of sheep for discretization

        # Discretize budget into 20 bins
        self.budget_bins = 20
        self.max_budget = 5000  # Maximum budget for discretization

        # Initialize state
        self.sheep_count = self.initial_sheep
        self.budget = initial_budget
        self.current_year = self.initial_year
        self.done = False

        # Define action and observation space
        self.action_space = spaces.Discrete(2)  # 0: Buy sheep, 1: Grow wheat
        self.observation_space = spaces.MultiDiscrete([self.budget_bins, self.max_sheep + 1, self.max_year + 1])

    def _get_discrete_state(self):
        discrete_budget = min(int(self.budget / (self.max_budget / self.budget_bins)), self.budget_bins - 1)
        discrete_sheep = min(self.sheep_count, self.max_sheep)
        return [discrete_budget, discrete_sheep, self.current_year]

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.budget = self.initial_budget
        self.sheep_count = self.initial_sheep
        self.current_year = self.initial_year
        self.done = False
        return self._get_discrete_state()

    def _move(self, action):
        if action == 0:  # Buy sheep
            if self.budget >= self.sheep_cost:
                self.budget -= self.sheep_cost
                self.sheep_count += 1
        elif action == 1:  # Grow wheat
            if self.budget >= self.wheat_cost:
                self.budget -= self.wheat_cost

        storm = np.random.random() < self.prob_storm

        wool_profit = self.sheep_count * 10
        wheat_profit = 0 if storm else 50
        self.budget += wool_profit + wheat_profit
        
        # Check for sheep reproduction
        if self.sheep_count >= 2:
            new_sheep = np.random.binomial(self.sheep_count // 2, self.prob_sheep)
            self.sheep_count += new_sheep

        reward = self.budget - self.initial_budget

        return reward

    def step(self, action):
        assert self.action_space.contains(action)
        
        self.current_year += 1
        reward = self._move(action)

        if self.budget <= 0 or self.current_year >= self.max_year:
            self.done = True

        return self._get_discrete_state(), reward, self.done, {}

    def render(self):
        print(f"Year: {self.current_year}, Budget: {self.budget}€, Sheep Count: {self.sheep_count}")

# Register the environment
register(
    id="TabularFarm-v0",
    entry_point=lambda: TabularFarmEnv(),
)

'''
        NOTES ON THE TABULAR METHOD IMPLEMENTED and main differences with the previous non-tabular method

Explanation of the discretization choices:

1. Budget: I've discretized the budget into 20 bins, with a maximum budget of 5000€. This provides a good balance between granularity and keeping the state space manageable.
2. Sheep count: I've capped the maximum number of sheep at 10 for the discrete state space. This should be sufficient for most scenarios while keeping the state space reasonable.
3. Year: The year is already discrete, so we keep it as is, ranging from 0 to 30.

Key differences and adaptations:

1. Removed the field_status from the state, as it's not necessary for decision-making in this simplified model.
2. Simplified the sheep reproduction model to use a binomial distribution based on the number of sheep pairs.
3. Removed the sheep_age dictionary, as we're not tracking individual sheep ages in this simplified model.
4. Changed the observation space to spaces.MultiDiscrete to represent the discrete state space.
5. Implemented _get_discrete_state() method to convert continuous state to discrete state.

This tabular environment should work well with Q-learning or other tabular reinforcement learning algorithms. The state space is now finite and discrete, making it suitable for a table-based approach.

'''


class ContinuousFarmEnv(gym.Env):

    def __init__(self, initial_sheep=0, initial_budget=2000, initial_year=0,
                 prob_storm=0.3, prob_sheep=0.1, sheep_cost=1000, wheat_cost=20):
        
        super().__init__()

        self.initial_budget = initial_budget
        self.initial_year = initial_year
        self.initial_sheep = initial_sheep
        self.prob_storm = prob_storm
        self.prob_sheep = prob_sheep
        self.sheep_cost = sheep_cost
        self.wheat_cost = wheat_cost

        self.max_year = 30
        self.max_sheep = 70
        self.max_budget = 50000

        # Initialize state
        self.sheep_count = self.initial_sheep
        self.budget = initial_budget
        self.current_year = self.initial_year
        self.done = False

        # Define action and observation space
        self.action_space = spaces.Discrete(2)  # 0: Buy sheep, 1: Grow wheat
        self.observation_space = spaces.Box(low=np.array([0, 0, 0]), 
                                            high=np.array([self.max_budget, self.max_sheep, self.max_year]),
                                            dtype=np.float32)

    def _get_state(self):
        return np.array([self.budget, self.sheep_count, self.current_year], dtype=np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.budget = self.initial_budget
        self.sheep_count = self.initial_sheep
        self.current_year = self.initial_year
        self.done = False
        return self._get_state()

    def _move(self, action):
        if action == 0:  # Buy sheep
            if self.budget >= self.sheep_cost:
                self.budget -= self.sheep_cost
                self.sheep_count += 1
        elif action == 1:  # Grow wheat
            if self.budget >= self.wheat_cost:
                self.budget -= self.wheat_cost

        storm = np.random.random() < self.prob_storm

        wool_profit = self.sheep_count * 10
        wheat_profit = 0 if storm else 50
        self.budget += wool_profit + wheat_profit
        
        # Check for sheep reproduction
        if self.sheep_count >= 2:
            new_sheep = np.random.binomial(self.sheep_count // 2, self.prob_sheep)
            self.sheep_count += new_sheep

        reward = self.budget - self.initial_budget

        return reward

    def step(self, action):
        assert self.action_space.contains(action)
        
        self.current_year += 1
        reward = self._move(action)

        if self.budget <= 0 or self.current_year >= self.max_year:
            self.done = True

        return self._get_state(), reward, self.done, {}

    def render(self):
        print(f"Year: {self.current_year}, Budget: {self.budget}€, Sheep Count: {self.sheep_count}")

#register(
#    id="TabularFarm-v0",
#    entry_point=lambda: TabularFarmEnv(),
#)

'''
        MAIN CHANGES FROM TABULAR TO CONTINUOUS METHODS

1. The class name is now ContinuousFarmEnv to reflect the continuous state space.
2. The observation_space is now defined as a Box space with continuous values for budget, sheep count, and year.
3. The _get_discrete_state method is replaced with _get_state, which returns the actual continuous values of the state variables.
4. The reset and step methods now return the continuous state representation. 

**add more detailed description
'''


#new version of the environment environment-v0 is available in wnvironment-v0.py
class VFAFarmEnv(gym.Env):

    def __init__(self, initial_sheep=0, initial_budget=2000, initial_year=0,
                 prob_storm=0.3, prob_sheep=0.1, sheep_cost=1000, wheat_cost=20):
        
        super().__init__()

        self.initial_budget = initial_budget
        self.initial_year = initial_year
        self.initial_sheep = initial_sheep
        self.prob_storm = prob_storm
        self.prob_sheep = prob_sheep
        self.sheep_cost = sheep_cost
        self.wheat_cost = wheat_cost

        self.max_year = 30
        self.max_sheep = 70
        self.max_budget = 5000

        # Initialize state
        self.sheep_count = self.initial_sheep
        self.budget = initial_budget
        self.current_year = self.initial_year
        self.done = False

        # Define action and observation space
        self.action_space = spaces.Discrete(2)  # 0: Buy sheep, 1: Grow wheat
        self.observation_space = spaces.Box(
            low=np.array([0, 0, 0]), 
            high=np.array([1, 1, 1]),
            dtype=np.float32
        )

    def _get_normalized_state(self):
        norm_budget = self.budget / self.max_budget
        norm_sheep = self.sheep_count / self.max_sheep
        norm_year = self.current_year / self.max_year
        return np.array([norm_budget, norm_sheep, norm_year], dtype=np.float32)

    
        

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.budget = self.initial_budget
        self.sheep_count = self.initial_sheep
        self.current_year = self.initial_year
        self.done = False
        return self._get_normalized_state()
    
    def _move(self, action):
        if action == 0:  # Buy sheep
            if self.budget >= self.sheep_cost:
                self.budget -= self.sheep_cost
                self.sheep_count += 1
        elif action == 1:  # Grow wheat
            if self.budget >= self.wheat_cost:
                self.budget -= self.wheat_cost

        storm = np.random.random() < self.prob_storm

        wool_profit = self.sheep_count * 10
        wheat_profit = 0 if storm else 50
        self.budget += wool_profit + wheat_profit
        
        # Check for sheep reproduction
        if self.sheep_count >= 2:
            new_sheep = np.random.binomial(self.sheep_count // 2, self.prob_sheep)
            self.sheep_count += new_sheep

        reward = self.budget - self.initial_budget

        return reward
    
    def step(self, action):
        assert self.action_space.contains(action)
        
        self.current_year += 1
        reward = self._move(action)

        if self.budget <= 0 or self.current_year >= self.max_year:
            self.done = True

        return self._get_normalized_state(), reward, self.done, {}

    def render(self):
        print(f"Year: {self.current_year}, Budget: {self.budget}€, Sheep Count: {self.sheep_count}")


register(
    id="VFAFarm-v0",
    entry_point=lambda: VFAFarmEnv(),
)
