import numpy as np
import gym
import pickle
from argparse import ArgumentParser

from YChain_rew_norm import YChain

parser = ArgumentParser(description="Parameters for the code")
parser.add_argument('--gamma', type=float, default=1, metavar='Gamma',
    help="Discount factor")
parser.add_argument('--episodes', type=int, default=10000, metavar='epi',
    help="Number of episodes")
parser.add_argument('--len', type=int, default=5,
    help="Chain Length")
parser.add_argument('--lr', type=float, default=0.1, metavar='lr_rate',
    help="Learning rate")
parser.add_argument('--decay', type=int, default=250, metavar='decay',
    help="Number of episodes before decaying learning rate")
parser.add_argument('--th', type=int, default=2500, metavar='decay_th',
    help="Number of episodes for decaying learning rate")
parser.add_argument('--init', type=int, default=0,
    help="initial values of states")
parser.add_argument('--beta_lr', type=float, default=0.001,
    help="learning rate to update beta net")
parser.add_argument('--cost', type=float, default=0.01,
    help="penalty for beta net")
parser.add_argument('--batch', type=bool, default=False,
    help="batch mode or online mode")

args = parser.parse_args()

def encode(num_state, state):
    temp_enc = np.zeros((1,num_state))
    temp_enc[0][state] = 1
    return temp_enc

class beta_net():

    def __init__(self, state_size, lr):
        self.state_size = state_size
        self.weights = np.random.randn(1,state_size)
        self.lr = lr
    
    def forward(self,inp):
        x = 1/(1+np.exp(-np.dot(self.weights, inp.T)))
        return x[0]

    def backward(self, memory):
        grad = 0
        for sig, target, v_tilde, v_tilde_prev, v, cost in memory:
            grad += (sig * (1 - sig)) * ((target - v_tilde) * (v_tilde_prev - v) + cost)
        self.grad = grad/len(memory)
        self.weights = self.weights - self.lr * self.grad

def printbeta(env, net):
    for i in range(env.n):
        net_inp = encode(env.n, i)
        print(net.forward(net_inp))
    print("**********")

all_values = []
for i in range(1):
    num_epi = args.episodes
    chain_len = args.len
    gamma = args.gamma
    lr = args.lr
    decay_epi = args.decay
    decay_th = args.th

    env = YChain(n=chain_len)
    value = np.array([0,0,1,-1])#args.init*np.ones(env.n)

    net = beta_net(env.n, args.beta_lr)

    memory = []

    for c_epi in range(num_epi):
        c_s = env.reset()
        done = False
        v_tilde_prev = 0
        rec_trace = np.zeros(env.n)
        all_td_err =[]
        while not done:

			# get current estimate
            net_inp = encode(env.n, c_s)
            beta_cs = net.forward(net_inp)
            v_tidle_curr = beta_cs * value[c_s] + (1 - beta_cs) * v_tilde_prev
            
			# update trace 
			curr_indicator = np.zeros(env.n)
            curr_indicator[c_s] = 1
            rec_trace = beta_cs * curr_indicator + (1 - beta_cs) * rec_trace
            
            action = env.action_space.sample()
            n_s, rew, done, _ = env.step(action)
            if not done:
                target = rew + gamma * value[n_s]
                td_error = target - v_tidle_curr
            else:
                target = rew
                td_error = target - v_tidle_curr

            if(td_error > 1/(1-args.gamma)):
                print("Update exploded")
            
            value = value + lr * td_error * rec_trace
            c_s = n_s
            v_tilde_prev = v_tidle_curr

            memory.append([beta_cs, target, v_tidle_curr, v_tilde_prev, value[c_s], args.cost])

            if args.batch == False:
                net.backward(memory)
                del memory[:]

        if (c_epi+1)%20 == 0 and args.batch == True:
            net.backward(memory)
            del memory[:]
            
        if (c_epi+1)%250 == 0 and (c_epi+1) < 2500:
            lr = lr/ 2

    all_values.append(value)

printbeta(env, net)
print("printing values")
print(value)

with open("learning_beta/norm_rew/value_chain_len_"+str(chain_len)+"_init_"+str(args.init)+".pkl", "wb") as f:
    pickle.dump(all_values, f)
