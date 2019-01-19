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
parser.add_argument('--print_every',       type=int, default=1000)
parser.add_argument('--alias_percentage',  type=float, default=0)
parser.add_argument('--lambda', type=float, default=0.5) # -1 for fully online
parser.add_argument('--online', type=int, default=1)
parser.add_argument('--lamb', type=float, default=0.9)
parser.add_argument('--return_type', type=str, default="MC")
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
    memory_beta  = []
    memory_value = []

    while not done:
        # fetch beta and value
        beta = b_net(state)   
        v    = v_net(state)      

        # build on smoothed estimate
        v_tilde = beta * v + (1. - beta) * v_tilde_prev
        
        # we are evaluating random policies. we pick a random action
        action = env.action_space.sample()
        next_state, reward, done, _ = env.step(action)

        # build target
        target = reward
        if not done: target += args.gamma * v_net(next_state)

        # update the trace
        if args.online: 
            v_net.update_trace(state, beta)

            # update state values
            td_error = target - v_tilde
            v_net.online_update(td_error)

            # update beta values
            b_net.update_logits([(state, beta, target, v_tilde, v_tilde_prev, v)])
        else:
            # store all required values for beta logits update
            memory_beta += [(state, beta, target, v_tilde, v_tilde_prev, v)]

            # store all required_values for value update
            memory_value += [(state, next_state, v_tilde, reward, beta)]

        state = next_state 
        v_tilde_prev = v_tilde - reward
        t += 1

    if not args.online:
        # time to update our values

        # get lambda or MC targets
        targets = v_net.get_targets(memory_value)

        # update values
        v_net.offline_update(memory_value, targets)
        
        # change the targets in the buffer
        for i, ((state, beta, target, v_tilde, v_tilde_prev, v), target) in enumerate(zip(memory_beta,targets)):
            memory_beta[i] = (state, beta, target, v_tilde, v_tilde_prev, v)

        b_net.update_logits(memory_beta)

    if (episode + 1) % args.print_every == 0:
        print('values\n'); print(v_net)
        print('betas \n'); print(b_net)