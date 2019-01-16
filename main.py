import sys 
import torch
import argparse
import numpy as np
from pydoc import locate
sys.path.append('./reinforcement-learning')

from agent import * 

# arguments
parser = argparse.ArgumentParser()
parser.add_argument('--env',        type=str,   default='cliff')
parser.add_argument('--lr',         type=float, default=0.001)
parser.add_argument('--beta_lr',    type=float, default=0.001)
parser.add_argument('--vc',         type=float, default=0.)
parser.add_argument('--gamma',      type=float, default=0.9)
parser.add_argument('--n_episodes', type=int,   default=100000)
parser.add_argument('--clip',       type=float, default=3)
parser.add_argument('--update_beta_every', type=int, default=1)
parser.add_argument('--print_every',       type=int, default=1000)
args = parser.parse_args()

# environment creation 
if 'cliff' in args.env.lower():
    from lib.envs.cliff_walking import CliffWalkingEnv
    env = CliffWalkingEnv()
    env.shape = (4, 12)
elif 'chain' in args.env.lower():
    raise NotImplementedError
else: 
    raise ValueError

# create estimators
v_net = V_net(args, env)
b_net = B_net(args, env)

for episode in range(args.n_episodes):
    done = False
    state = env.reset()
    v_tilde_prev = 0.
    t = 0    

    # buffer for beta updates
    memory = []

    while not done:
        # fetch beta and value
        beta = b_net(state)   
        v    = v_net(state)      

        # build on smoothed estimate
        v_tilde = beta * v + (1 - beta) * v_tilde_prev
        
        # update the trace
        v_net.update_trace(state, beta)

        # we are evaluating random policies. we pick a random action
        action = env.action_space.sample()
        next_state, reward, done, _ = env.step(action)

        # build target
        target = reward
        if not done: 
            # TODO: do we want to support other targets ?
            target += args.gamma * v_net(next_state)

        # update state values
        td_error = target - v_tilde
        v_net.update_values(td_error)

        # store all required values for beta logits update
        memory += [(state, beta, target, v_tilde, v_tilde_prev, v)]

        # update beta values
        if (t + 1) % args.update_beta_every == 0: 
            b_net.update_logits(memory)
            memory = []
                
        state = next_state 
        v_tilde_prev = v_tilde
        t += 1


    if (episode + 1) % args.print_every == 0: 
        print('values\n', v_net)
        print('betas \n', b_net)
       
