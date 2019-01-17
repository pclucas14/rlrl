import numpy as np
from utils import pprint


'''
our agent
'''

class Aliaser(object):
    def __init__(self, args, env):
        
        # aliasing setup for POMDP
        assert args.alias_percentage == 0 or 'chain' in args.env.lower()
        
        self.state_mapping = {i:i for i in range(env.observation_space.n)}
        chain_length = (env.observation_space.n - 1) // 3
        no_aliased_states = int(args.alias_percentage / 100. * chain_length)
        self.aliased_indices = np.random.choice(chain_length, no_aliased_states)
        self.aliased_indices += chain_length + 1
        
        for aliased_index in self.aliased_indices:
            self.state_mapping[aliased_index] = \
                    aliased_index + chain_length

    def __call__(self, index):
        return self.state_mapping[index]


class V_net(object):
    def __init__(self, args, env, aliaser):
        self.trace  = np.zeros((env.observation_space.n, ))    
        self.values = np.zeros((env.observation_space.n, )) + args.init_value 
        self.args   = args
        self.env    = env
        self.ali    = aliaser

        # mask out aliased indices for clarity when printing
        self.trace[self.ali.aliased_indices] = -1
    
    def update_trace(self, state, beta):
        self.trace = (1. - beta) * self.trace
        self.trace[self.ali(state)] += beta

    def reset_trace(self):
        self.trace *= 0. 

    def update_values(self, td_error):
        self.values += self.args.lr * td_error * self.trace

    # fetch state value
    def __call__(self, x):
        return self.values[self.ali(x)]

    def __str__(self):
       values = self.values
       values[self.ali.aliased_indices] = -9.99
       return pprint(values.reshape(self.env.shape))

     
class B_net(object):
    def __init__(self, args, env, aliaser):
        self.b_logits = np.zeros((env.observation_space.n, ))
        self.args     = args
        self.env      = env
        self.ali      = aliaser

    def stable_sigmoid(self, x):
        return np.exp(-np.logaddexp(0, -x)) 

    def update_logits(self, memory):
        for (state, sig, target, v_tilde, v_tilde_prev, v) in memory:
            grad = sig * (1 - sig) * ((v_tilde - target) * (v - v_tilde_prev) + self.args.vc)
            self.b_logits[self.ali(state)] -= self.args.beta_lr * grad 

    # fetch beta value
    def __call__(self, x):
        return self.stable_sigmoid(self.b_logits)[self.ali(x)]

    def __str__(self):
        values = self.stable_sigmoid(self.b_logits)
        values[self.ali.aliased_indices] = -9.99
        return pprint(values.reshape(self.env.shape))
        
