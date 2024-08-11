import numpy as np
import gymnasium as gym
from gymnasium import spaces
from gymnasium.envs.registration import register

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
        self.wheat_grown = 0

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
    
    def scale_reward(self, reward):
        min_reward = -self.initial_budget  # Worst case: losing all money
        max_reward = self.max_budget - self.initial_budget  # Best case: reaching max budget
        return (reward - min_reward) / (max_reward - min_reward)
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.budget = self.initial_budget
        self.sheep_count = self.initial_sheep
        self.current_year = self.initial_year
        self.done = False
        self.wheat_grown = 0
        return self._get_normalized_state(), {}
    
    def _move(self, action):
        if action == 0:  # Buy sheep
            if self.budget >= self.sheep_cost:
                self.budget -= self.sheep_cost
                self.sheep_count += 1
        elif action == 1:  # Grow wheat
            if self.budget >= self.wheat_cost:
                self.budget -= self.wheat_cost
                self.wheat_grown += 1

        storm = np.random.random() < self.prob_storm

        wool_profit = self.sheep_count * 10
        wheat_profit = 0 if storm else self.wheat_grown * 50
        self.budget += wool_profit + wheat_profit
        
        # Check for sheep reproduction
        if self.sheep_count >= 2:
            new_sheep = np.random.binomial(self.sheep_count // 2, self.prob_sheep)
            self.sheep_count += new_sheep

        reward = self.budget - self.initial_budget

        return reward, storm
    
    def step(self, action):
        assert self.action_space.contains(action)
        
        self.current_year += 1
        raw_reward, storm = self._move(action)

        if self.budget < 0:
            self.done = True
            raw_reward = -1000

        elif self.current_year >= self.max_year:
            self.done = True

        scaled_reward = self.scale_reward(raw_reward) 

        info = {
            "year": self.current_year,
            "budget": self.budget,
            "sheep_count": self.sheep_count,
            "wheat_grown": self.wheat_grown,
            "storm_occurred": storm,
            "raw_reward": raw_reward 
        }

        return self._get_normalized_state(), scaled_reward, self.done, info

    def render(self):
        print(f"Year: {self.current_year}, Budget: {self.budget}€, Sheep Count: {self.sheep_count}")


register(
    id="VFAFarm-v0.0.1",
    entry_point=lambda: VFAFarmEnv(),
)

#v0.1 differ from v0 only for the fact that during the training can access all the variables of the 
# environment and so is able to plot all the variable and I also have a new variable for the wheat eventually grown