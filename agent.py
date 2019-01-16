import numpy as np
from utils import pprint

'''
our agent
'''

class V_net(object):
    def __init__(self, args, env):
        self.trace  = np.zeros((env.observation_space.n, ))    
        self.values = np.zeros((env.observation_space.n, ))   
        self.args   = args
        self.env    = env
 
    def update_trace(self, state, beta):
        self.trace = (1 - beta) * self.trace
        self.trace[state] += beta 

    def update_values(self, td_error):
        self.values += self.args.lr * td_error * self.trace

    # fetch state value
    def __call__(self, x):
        return self.values[x]

    def __str__(self):
       return pprint(self.values.reshape(self.env.shape))

     
class B_net(object):
    def __init__(self, args, env):
        self.b_logits = np.zeros((env.observation_space.n, ))
        self.args     = args
        self.env      = env

    def stable_sigmoid(self, x):
        return np.exp(-np.logaddexp(0, -x)) 

    def clip(self, x):
        return max(-self.args.clip, min(self.args.clip, x))
    
    def update_logits(self, memory):
        for (state, sig, target, v_tilde, v_tilde_prev, v) in memory:
            grad = sig * (1 - sig) * ((v_tilde - target) * (v - v_tilde_prev) + self.args.vc)
            self.b_logits[state] -= self.args.beta_lr * self.clip(grad)

    # fetch beta value
    def __call__(self, x):
        return self.stable_sigmoid(self.b_logits)[x]

    def __str__(self):
        return pprint(self.stable_sigmoid(self.b_logits).reshape(self.env.shape))
        
        

