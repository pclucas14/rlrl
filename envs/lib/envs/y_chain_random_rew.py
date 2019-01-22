import gym
from gym import spaces
from gym.utils import seeding
from gym.envs.registration import register
import numpy as np

class YChain():
    def __init__(self, n=5, var=1):
        self.len_chain = n #length of one chain
        self.n = n*3 + 1 #length of MDP
        self.state = 0  # Start at beginning of the chain
        self.action_space = spaces.Discrete(2) #number of actions - 2 - [0: go left, 1: go right]
        self.pos_reward = +1
        self.neg_reward = -1
        self.observation_space = spaces.Discrete(self.n)
        self.bottle_neck = n #bottleneck state - that connects the 3 chains
        self.random_rew = np.random.normal(0,var)

    def seed(self, seed=None):
        np.random.seed(seed)
        return [seed]
        #self.np_random, seed = seeding.np_random(seed)
        #return [seed]

    def step(self, action):
        '''
        takes an action as an argument and returns the next_state, reward, done, info.
        '''
        assert self.action_space.contains(action)
        reward = 0 + self.random_rew
        done = False
        
        # deciding on the next chain to switch if in the bottleneck state
        if self.state == self.bottle_neck:
            if not action:
                self.state += 1
            else:
                self.state = self.len_chain * 2 + 1
                
        # keep moving forward in the chain if not in the bottleneck state irrespective of the action
        else:
            
            # if in next transition is terminal state, give out reward
            if (self.state == self.len_chain * 2) or (self.state == self.len_chain * 3):
                reward = (self.pos_reward+self.random_rew if self.state == self.len_chain*2 else self.neg_reward+self.random_rew)
                done = True
                self.state = None
                
            else:
                self.state += 1
        return self.state, reward, done, {}

    def reset(self):
        '''
        transitions back to first state
        '''
        self.state = 0
        return self.state
    
register(
    id='YChain-v0',
    entry_point='ychain:YChain',
    timestep_limit=20000,
    reward_threshold=1,
)