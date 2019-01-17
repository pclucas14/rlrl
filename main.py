import sys 
import torch
import argparse
import numpy as np
from pydoc import locate
sys.path.append('./envs')

from agent import * 

# arguments
parser = argparse.ArgumentParser()
parser.add_argument('--env',        type=str,   default='chain')
parser.add_argument('--lr',         type=float, default=0.1)
parser.add_argument('--beta_lr',    type=float, default=0.05)
parser.add_argument('--init_value', type=float, default=0)
parser.add_argument('--vc',         type=float, default=0.)
parser.add_argument('--gamma',      type=float, default=1.)
parser.add_argument('--n_episodes', type=int,   default=100000)
parser.add_argument('--clip',       type=float, default=999)
parser.add_argument('--update_beta_every', type=int, default=1)
parser.add_argument('--print_every',       type=int, default=1000)
parser.add_argument('--alias_percentage',  type=float, default=0)
args = parser.parse_args()

# environment creation 
if 'cliff' in args.env.lower():
    from lib.envs.cliff_walking import CliffWalkingEnv
    env = CliffWalkingEnv()
    env.shape = (4, 12)
elif 'chain' in args.env.lower():
    from lib.envs.y_chain import YChain
    env = YChain()
    env.shape = (1, -1)
else: 
    raise ValueError

# create estimators
aliaser = Aliaser(args, env) 
v_net  = V_net(args, env, aliaser)
b_net  = B_net(args, env, aliaser)

for episode in range(args.n_episodes):
    done = False
    state = env.reset()
    v_net.reset_trace()
    v_tilde_prev = 0.
    t = 0    

    # buffer for beta updates
    memory = []

    while not done:
        # fetch beta and value
        beta = b_net(state)   
        v    = v_net(state)      

        # build on smoothed estimate
        v_tilde = beta * v + (1. - beta) * v_tilde_prev
        
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
        v_tilde_prev = v_tilde - reward
        t += 1

    if (episode + 1) % args.print_every == 0:
        print('values\n'); print(v_net)
        print('betas \n'); print(b_net)
       
