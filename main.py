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
parser.add_argument('--gamma',      type=float, default=0.9)
parser.add_argument('--n_episodes', type=int,   default=100000)
parser.add_argument('--clip',       type=float, default=999)
parser.add_argument('--print_every',       type=int, default=1000)
parser.add_argument('--alias_count',  type=int, default=0)
parser.add_argument('--online', type=int, default=1)
parser.add_argument('--lamb', type=float, default=0.9)
parser.add_argument('--return_type', type=str, default="MC")
parser.add_argument('--recurrent', type=int, default=1)
parser.add_argument('--seed', type=int, default=42)
args = parser.parse_args()

np.random.seed(args.seed)

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

if args.recurrent:
    b_net  = B_net(args, env, aliaser)

MSVE = []

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
        # fetch beta
        if args.recurrent:
            beta = b_net(state)

        # settin beta=1 to run vanilla TD
        if not args.recurrent:
            beta = 1

        # fetch value
        v    = v_net(state)      

        # build on smoothed estimate
        v_tilde = beta * v + (1. - beta) * v_tilde_prev
        
        # we are evaluating random policies. we pick a random action
        # action = env.action_space.sample()
        action = np.random.randint(0,2)
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
            if args.recurrent:
                b_net.update_logits([(state, beta, target, v_tilde, v_tilde_prev, v)])
        else:
            # store all required values for beta logits update
            memory_beta += [(state, beta, target, v_tilde, v_tilde_prev, v)]

            # store all required_values for value update
            memory_value += [(state, next_state, v_tilde, reward, beta)]

        if state == 8:
            error = (v_tilde - 0.81)**2
            MSVE.append(error)
        if state == 13:
            error = (v_tilde + 0.81)**2
            MSVE.append(error)

        if episode+10 > args.n_episodes:
            v_net.vbeta.append(v_tilde)

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
        if args.recurrent:
            for i, ((state, beta, target, v_tilde, v_tilde_prev, v), a_target) in enumerate(zip(memory_beta,targets)):
                memory_beta[i] = (state, beta, a_target, v_tilde, v_tilde_prev, v)
            b_net.update_logits(memory_beta)

    
    if (episode+10) > args.n_episodes:
        print("\nv-beta"); print(np.around(np.array(v_net.vbeta), decimals=2)); print("\n")
        v_net.vbeta = []
    

    if (episode + 1) % args.print_every == 0:
        print("****************")
        print('values\n'); print(v_net)
        if args.recurrent:
            print('betas \n'); print(b_net)

import pickle
if not args.recurrent:
    with open('./pomdp_exp/vanilla/'+str(args.seed)+'.pkl', 'wb') as f:
        pickle.dump(MSVE, f)
else:
    with open('./pomdp_exp/recurrent/'+str(args.seed)+'.pkl', 'wb') as f:
        pickle.dump(MSVE, f)
'''
import matplotlib.pyplot as plt
plt.plot(range(args.n_episodes), MSVE)
plt.show()
'''