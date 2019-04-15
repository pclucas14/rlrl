import sys 
import torch
import argparse
import matplotlib.pyplot as plt
import numpy as np
from pydoc import locate
sys.path.append('./envs')
import math
from agent import * 

# arguments
parser = argparse.ArgumentParser()
parser.add_argument('--env',        type=str,   default='chain')
parser.add_argument('--lr',         type=float, default=0.1)
parser.add_argument('--beta_lr',    type=float, default=0)
parser.add_argument('--init_value', type=float, default=0)
parser.add_argument('--vc',         type=float, default=0.01)
parser.add_argument('--gamma',      type=float, default=0.99)
parser.add_argument('--n_steps', type=int,   default=5000)
parser.add_argument('--clip',       type=float, default=999)
parser.add_argument('--print_every',       type=int, default=1000)
parser.add_argument('--alias_percentage',  type=float, default=0)
#parser.add_argument('--lambd', type=float, default=1) # -1 for fully online
parser.add_argument('--online', type=int, default=1)
parser.add_argument('--lambd', type=float, default=0)
parser.add_argument('--return_type', type=str, default="MC")
parser.add_argument('--rep',type=int,default=10) # number of repetition
args = parser.parse_args()


# environment creation 
if 'cliff' in args.env.lower():
    from lib.envs.cliff_walking import CliffWalkingEnv
    env = CliffWalkingEnv()
    env.shape = (4, 12)
elif 'ring' in args.env.lower():
    from lib.envs.ring_chain import Ring
    env = Ring()
    env.shape = (1, -1)
elif 'chain' in args.env.lower():
    from lib.envs.y_chain import YChain
    env = YChain()
    env.shape = (1, -1)

else: 
    raise ValueError
# create estimators
length_episode = 20
tmp_opt = [0.00 , 8.53, 8.39, 8.18,8.01 , 7.74, 7.57,7.31, 6.92,0.00]
for idx_2,(beta_val,lambd) in enumerate([(1,0.9),(0.75,0.9),(0.5,0.9)]):
    error_list = []
    for idx_1, lr in enumerate(np.linspace(0.1, 0.7, num=10)):

        #print("Param: ",str(lr),"  ",str(beta_lr))
        #args.beta_lr = beta_lr
        args.lr = lr
        args.lambd = lambd
        error = 0
        for r in range(args.rep):
            aliaser = Aliaser(args, env)
            v_net = V_net(args, env, aliaser, args.lambd, args.gamma)
            b_net = B_net(args, env, aliaser)

            done = False
            state = env.reset()
            v_net.reset_trace()
            v_tilde_prev = 0.
            t = 0
            beta = beta_next = beta_val
            # buffer for beta updates
            memory_beta = []
            memory_value = []
            for step in range(args.n_steps):

                # we are evaluating random policies. we pick a random action
                if 'ring' in args.env.lower():
                    action = np.random.choice([0,1],p=[0.7,0.3])
                else:
                    action = env.action_space.sample()
                next_state, reward, done, _ = env.step(action)
                #print("State: ",state," Next state ",next_state," V_tilde: ", v_tilde)
                # fetch beta and value
                if args.beta_lr != 0: ## If we set learning rate to beta to 0 then we go back to TD
                    beta = b_net(state)
                    beta_next = b_net(next_state)
                v  = v_net(state)
                # build on smoothed estimate
                v_tilde = (beta * v + (1. - beta) * v_tilde_prev)
                # build target
                target = reward
                v_tilde_next = beta_next * v_net(next_state) + (1-beta_next) * ((v_tilde - reward)/args.gamma)
                if not done: target += args.gamma * v_tilde_next

                # update the trace
                # update the trace
                v_net.update_trace(state, beta)

                # update state values
                td_error = target - v_tilde
                v_net.online_update(td_error)

                # update beta values
                b_net.update_logits([(state, beta, target, v_tilde, v_tilde_prev, v)])

                state = next_state
                v_tilde_prev = (v_tilde - reward)/args.gamma
                error += np.sum(np.square(v_net.values - tmp_opt))
                if error > 10000000 or math.isnan(error):
                    error = 10000000
                #if (step + 1) % args.print_every == 0:
                #    print('values\n'); print(v_net)
                #    print('betas \n'); print(b_net)

        error_list.append(error)
    plt.plot(error_list,label=str(beta_val)+str(lr))
    print(error_list)
plt.legend()
plt.show()