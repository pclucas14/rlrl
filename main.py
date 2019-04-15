import sys
from comet_ml import Experiment
#from comet_ml import OfflineExperiment
from comet_ml import Optimizer
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
parser.add_argument('--est-beta',   type=float, default=0)
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
parser.add_argument('--disable-log', action='store_true', default=False)
parser.add_argument('--rep',type=int,default=10) # number of repetition
args = parser.parse_args()

optimizer = Optimizer("HFFoR5WtTjoHuBGq6lYaZhG0c")
#est_beta integer [0, 0] [0]
#beta_lr real [0.1, 0.1] [0.2]
#
params = """
lambd real [0, 1] [0]
lr real [0.1,1] [0.2]
beta_val real [0, 1] [1]
"""

optimizer.set_params(params)



def fit(args,suggestion):
    #args.est_beta = suggestion["est_beta"]
    args.est_beta = 0
    beta_val =  suggestion["beta_val"]
    args.lambd = suggestion["lambd"]
    args.lr = suggestion["lr"]
    #args.beta_lr = suggestion["beta_lr"]
    args.beta_lr = 0
    length_episode = 20
    tmp_opt = [0.00, 8.53, 8.39, 8.18, 8.01, 7.74, 7.57, 7.31, 6.92, 0.00]
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
    error_list = []
    for r in range(args.rep):
        aliaser = Aliaser(args, env)
        v_net = V_net(args, env, aliaser, args.lambd, args.gamma)
        b_net = B_net(args, env, aliaser)
        error = 0
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
            if args.est_beta == 1: ## If we set learning rate to beta to 0 then we go back to TD
                beta = b_net(state)
                beta_next = b_net(next_state)
            v  = v_net(state)
            # build on smoothed estimate
            v_tilde = (beta * v + (1. - beta) * v_tilde_prev)
            # build target
            target = reward
            #v_tilde_next = beta_next * v_net(next_state) + (1-beta_next) * ((v_tilde - reward)/args.gamma)
            if not done: target += args.gamma * v_net(next_state)
            try:
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
                error = np.abs(v_tilde - tmp_opt[state])
                if error < -10 or math.isnan(error) or math.isinf(error):
                    error = -10
                error_list.append(error)
            except:
                error_list.append(-10)


            #if (step + 1) % args.print_every == 0:
            #    print('values\n'); print(v_net)
            #    print('betas \n'); print(b_net)
    error = np.array(error_list).mean()*(-1)
    if error < -10  or math.isnan(error) or math.isinf(error):
        error = -10
    return error

while True:
    suggestion = optimizer.get_suggestion()
    experiment = Experiment("HFFoR5WtTjoHuBGq6lYaZhG0c",project_name = "ring-recurrent",workspace="pierthodo")
    score = fit(args,suggestion)
    suggestion.report_score("accuracy", score)
