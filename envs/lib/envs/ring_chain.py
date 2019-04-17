import gym
from gym import spaces
from gym.utils import seeding
from gym.envs.registration import register


class Ring():
    def __init__(self, n=9):
        self.n = n  # length of  ring
        self.state = n//2  # Start at beginning of the chain
        self.action_space = spaces.Discrete(2)  # number of actions - 2 - [0: go left, 1: go right]
        self.pos_reward = 1
        self.neg_reward = -1
        self.observation_space = spaces.Discrete(self.n)
        self.seed()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        '''
        takes an action as an argument and returns the next_state, reward, done, info.
        '''
        assert self.action_space.contains(action)
        reward = 0
        done = False
        if action == 1:
            self.state += 1
        else:
            self.state -= 1
        if (self.state == 0 or self.state == self.n - 1):
            reward =(self.pos_reward if self.state == 0 else self.neg_reward)
            #done = True
            self.state = self.n //2

        return self.state, reward, done, {}

    def reset(self):
        '''
        transitions back to first state
        '''
        self.state = self.n//2
        return self.state


register(
    id='Ring-chain-v0',
    entry_point='ring:Ring',
    timestep_limit=20000,
    reward_threshold=1,
)




