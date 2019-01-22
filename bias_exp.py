import sys 
import os
import torch
import argparse
import numpy as np
from pydoc import locate
sys.path.append('./envs')

from agent import * 

# arguments
parser = argparse.ArgumentParser()
parser.add_argument('--env',        type=str,   default='chain')
parser.add_argument('--var',        type=float, default=0)
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
parser.add_argument('--return_type', type=str, default="Lambda")
parser.add_argument('--learning', type=int, default=1)
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--beta_init', type=float, default=0.5)
args = parser.parse_args()

np.random.seed(args.seed)
print(args.seed)

# environment creation 
if 'cliff' in args.env.lower():
    from lib.envs.cliff_walking import CliffWalkingEnv
    env = CliffWalkingEnv()
    env.shape = (4, 12)
elif 'chain' in args.env.lower():
    if args.var == 0:
        from lib.envs.y_chain import YChain
        env = YChain()
    else:
        from lib.envs.y_chain_random_rew import YChain
        env = YChain(var=args.var)
    env.shape = (1, -1)
else: 
    raise ValueError


# create estimators
aliaser = Aliaser(args, env) 
v_net  = V_net(args, env, aliaser)

if args.learning:
    b_net  = B_net(args, env, aliaser)

MSVE = []
V_optimal = {i:j for i, j in enumerate([0.,0.,0.,0.,0.,0.,0.67,0.73,0.81,0.90,1.,-0.67,-0.73,-0.81,-0.90,-1.])}

for episode in range(args.n_episodes):
    done = False
    state = env.reset()
    v_net.reset_trace()
    v_tilde_prev = 0.
    t = 0    
    # buffer for beta updates
    memory_beta  = []
    memory_value = []

    error = 0
    while not done:
        # fetch beta
        if args.learning:
            beta = b_net(state)

        # settin beta=1 to run vanilla TD
        if not args.learning:
            beta = args.beta_init

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
            if args.learning:
                b_net.update_logits([(state, beta, target, v_tilde, v_tilde_prev, v)])
        else:
            # store all required values for beta logits update
            memory_beta += [(state, beta, target, v_tilde, v_tilde_prev, v)]

            # store all required_values for value update
            memory_value += [(state, next_state, v_tilde, reward, beta)]

        #if state > 5:
        error += (V_optimal[state] - v_net(state))**2
        
        state = next_state 
        v_tilde_prev = v_tilde - reward
        t += 1

    MSVE.append(np.sum(error))

    if not args.online:
        # time to update our values

        # get lambda or MC targets
        targets = v_net.get_targets(memory_value)

        # update values
        v_net.offline_update(memory_value, targets)
        
        # change the targets in the buffer
        if args.learning:
            for i, ((state, beta, target, v_tilde, v_tilde_prev, v), a_target) in enumerate(zip(memory_beta,targets)):
                memory_beta[i] = (state, beta, a_target, v_tilde, v_tilde_prev, v)
            b_net.update_logits(memory_beta)
    

    if (episode + 1) % args.print_every == 0:
        print("****************")
        print('values\n'); print(v_net)
        if args.learning:
            print('betas \n'); print(b_net)

import pickle
if args.var == 0:
    if not args.learning:
        with open('./bias_exp/fixed/'+str(args.seed)+'.pkl', 'wb') as f:
            pickle.dump(MSVE, f)
    else:
        with open('./bias_exp/learning/'+str(args.seed)+'.pkl', 'wb') as f:
            pickle.dump(MSVE, f)

else:
    if not os.path.exists('./var_exp/'+str(args.var)):
        os.makedirs('./var_exp/'+str(args.var))
    if not args.learning and args.beta_init != 1:
        if not os.path.exists('./var_exp/'+str(args.var)+'/fixed/'):
            os.makedirs('./var_exp/'+str(args.var)+'/fixed/')
        with open('./var_exp/'+str(args.var)+'/fixed/'+str(args.seed)+'.pkl', 'wb') as f:
            pickle.dump(MSVE, f)
    elif not args.learning and args.beta_init == 1.0:
        if not os.path.exists('./var_exp/'+str(args.var)+'/TD/'):
            os.makedirs('./var_exp/'+str(args.var)+'/TD/')
        with open('./var_exp/'+str(args.var)+'/TD/'+str(args.seed)+'.pkl', 'wb') as f:
            pickle.dump(MSVE, f)
    else:
        if not os.path.exists('./var_exp/'+str(args.var)+'/learning/'):
            os.makedirs('./var_exp/'+str(args.var)+'/learning/')
        with open('./var_exp/'+str(args.var)+'/learning/'+str(args.seed)+'.pkl', 'wb') as f:
            pickle.dump(MSVE, f)

'''
import matplotlib.pyplot as plt
plt.plot(range(args.n_episodes), MSVE)
plt.show()
'''