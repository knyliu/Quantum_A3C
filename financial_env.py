import numpy as np

class SimpleSpace:
    def __init__(self, shape=None, n=None):
        self.shape = shape
        self.n = n

class FinancialTradingEnv:
    """A minimal trading environment for RL experiments."""

    def __init__(self, prices, initial_balance=0.0):
        self.prices = np.asarray(prices, dtype=np.float32)
        self.initial_balance = initial_balance
        self.action_space = SimpleSpace(n=3)  # hold, buy, sell
        self.observation_space = SimpleSpace(shape=(1,))
        self.reset()

    def reset(self):
        self.t = 0
        self.balance = self.initial_balance
        self.position_price = None
        return np.array([self.prices[self.t]], dtype=np.float32)

    def step(self, action):
        reward = 0.0
        terminated = False
        truncated = False

        price = self.prices[self.t]
        if action == 1:  # buy
            if self.position_price is None:
                self.position_price = price
        elif action == 2:  # sell
            if self.position_price is not None:
                reward = price - self.position_price
                self.balance += reward
                self.position_price = None

        self.t += 1
        if self.t >= len(self.prices):
            terminated = True
            obs = np.array([self.prices[-1]], dtype=np.float32)
        else:
            obs = np.array([self.prices[self.t]], dtype=np.float32)

        return obs, reward, terminated, truncated, {}

def generate_price_series(length=100, seed=None):
    rng = np.random.default_rng(seed)
    steps = rng.normal(loc=0.0, scale=1.0, size=length)
    return np.cumsum(steps) + 100.0
