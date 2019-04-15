import numpy as np
from utils import pprint


'''
our agent
'''

class Aliaser(object):
    def __init__(self, args, env):
        
        # aliasing setup for POMDP
        assert args.alias_percentage == 0 or 'chain' in args.env.lower()

        if 'ring' in args.env.lower():
            self.state_mapping = {i: i for i in range(env.observation_space.n)}
            chain_length = (env.observation_space.n - 1) // 2
            no_aliased_states = int(args.alias_percentage / 100. * chain_length)
            self.aliased_indices = np.random.choice(chain_length, no_aliased_states)
            self.aliased_indices += 1
        #    a = 0
            for aliased_index in self.aliased_indices:
                self.state_mapping[aliased_index] = aliased_index
        else:
            self.state_mapping = {i:i for i in range(env.observation_space.n)}
            chain_length = (env.observation_space.n - 1) // 3
            no_aliased_states = int(args.alias_percentage / 100. * chain_length)
            self.aliased_indices = np.random.choice(chain_length, no_aliased_states)
            self.aliased_indices += chain_length + 1

            for aliased_index in self.aliased_indices:
                self.state_mapping[aliased_index] = aliased_index + chain_length

    def __call__(self, index):
        return self.state_mapping[index]


class V_net(object):
    def __init__(self, args, env, aliaser,lambd,gamma):
        self.trace  = np.zeros((env.observation_space.n, ))
        self.beta_trace  = np.zeros((env.observation_space.n, ))

        self.values = np.zeros((env.observation_space.n, )) + args.init_value 
        self.args   = args
        self.env    = env
        self.ali    = aliaser
        self.lambd = lambd
        self.gamma = gamma
        # mask out aliased indices for clarity when printing
        self.trace[self.ali.aliased_indices] = -1
    
    def update_trace(self, state, beta):
        self.beta_trace = (1. - beta) * self.beta_trace
        self.beta_trace[self.ali(state)] += beta
        self.trace = self.gamma * self.lambd * self.trace + self.beta_trace
        #self.trace = self.beta_trace

    def reset_trace(self):
        self.beta_trace *= 0
        self.trace *= 0. 

    def online_update(self, td_error):
        self.values += self.args.lr * td_error * self.trace

    def calc_lambda_return(self, G):

        # calculate lambda returns from a list of n-step returns
        out = 0
        if len(G) > 1:
            for idx, ret in enumerate(G[:-1]):
                out += (1-self.args.lamb)* self.args.lamb**idx * ret
            out += self.args.lamb**(idx+1) * G[-1]
            return out
        else:
            return G[0]

    def get_targets(self, memory):

        # calculate the targets whether MC or lambda based

        # MC return calculation
        if self.args.return_type == "MC":
            MC_returns = []
            G = 0
            for (state, _, v_tilde, reward, beta) in reversed(memory):
                G = self.args.gamma * G + reward
                MC_returns.append(G)
            return MC_returns[::-1]

        # lambda return calculation
        elif self.args.return_type == "Lambda":
            Lambda_returns = []
            rew = []
            val = []
            R = 0
            for idx, (state, next_state, v_tilde, reward, beta) in enumerate(memory):
                R = R + self.args.gamma**idx * reward
                rew.append(R)
                if next_state is not None:
                    val.append(self.args.gamma**(idx+1) * self.values[next_state])
                else:
                    val.append(0)
            G = [i+j for i,j in zip(rew, val)]

            Lambda_returns.append(self.calc_lambda_return(G))

            for i in range(1, len(memory)):
                rew_temp = memory[i-1][3]
                G = list((np.array(G[1:]) - rew_temp)/self.args.gamma)
                Lambda_returns.append(self.calc_lambda_return(G))

            return Lambda_returns

        else:
            raise ValueError

    def offline_update(self, memory, targets):
        for (state, next_state, v_tilde, reward, beta), target in zip(memory, targets):
            self.update_trace(state, beta)
            # update state values
            td_error = target - v_tilde
            self.online_update(td_error)        

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
        
