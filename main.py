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
import matplotlib.pyplot as plt
from agent import *
from pprint import pprint


# arguments
parser = argparse.ArgumentParser()
parser.add_argument('--env',        type=str,   default='chain')
parser.add_argument('--lr',         type=float, default=0.25)
parser.add_argument('--beta-lr',    type=float, default=0)
parser.add_argument('--est-beta',   type=float, default=0)
parser.add_argument('--init_value', type=float, default=0)
parser.add_argument('--vc',         type=float, default=0.01)
parser.add_argument('--gamma',      type=float, default=0.9)
parser.add_argument('--n-steps', type=int,   default=5000)
parser.add_argument('--clip',       type=float, default=999)
parser.add_argument('--print-every',       type=int, default=50)
parser.add_argument('--print-image',type=int,default=100)
parser.add_argument('--decay-lr',type=float,default=0)
parser.add_argument('--alias-percentage',  type=float, default=0)
#parser.add_argument('--lambd', type=float, default=1) # -1 for fully online
parser.add_argument('--online', type=int, default=1)
parser.add_argument('--lambd', type=float, default=0)
parser.add_argument('--return_type', type=str, default="MC")
parser.add_argument('--disable-log', action='store_true', default=False)
parser.add_argument('--rep',type=int,default=10) # number of repetition
parser.add_argument('--beta-val',type=float,default=1)
parser.add_argument('--hyper-comet',type=int,default=0)
args = parser.parse_args()

optimizer = Optimizer("HFFoR5WtTjoHuBGq6lYaZhG0c")
#est_beta integer [0, 0] [0]
#beta_lr real [0.1, 0.1] [0.2]
#
params = """
lambd real [0,1] [1]
beta_val real [0.5,1] [1]
"""
dic_param = []

lambd_list = [0,0.2,0.4,0.6,0.8,1]
beta_val_list = [0.5,0.6,0.7,0.8,0.9,1]
lr_list = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1]
#lambd_list = [1]
#beta_val_list = [1]
#lr_list = [0.1,0.2]
for l in lambd_list:
    for b in beta_val_list:
        for l in lr_list:
            dic_param.append({"lambd":l,"beta_val":b,"lr":l})

optimizer.set_params(params)



def fit(args,suggestion,experiment):
    if suggestion != "":
        #args.est_beta = suggestion["est_beta"]
        args.est_beta = 0
        args.beta_val =  suggestion["beta_val"]
        #args.beta_val = 1

        args.lambd = suggestion["lambd"]
        #args.lambd = 0
        #args.lr = suggestion["lr"]
        #args.beta_lr = suggestion["beta_lr"]
        args.beta_lr = 0
    else:
        args.rep = 1
    #args.beta_lr = 0

    length_episode = 20
    #experiment.log_parameters(vars(args))
    tmp_opt = [0., 2.6785975,  2.38961887 ,2.12787366, 1.88752481, 1.55648397,0.92376508 ,0.06853928 ,0.]


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
    error_list_rep = []
    error_list_aliased = []
    value_list = []
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
        beta = beta_next = args.beta_val
        error_list = []
        for step in range(args.n_steps):
            args.lr *= (1. / (1. + args.decay_lr * step))

            # we are evaluating random policies. we pick a random action
            if 'ring' in args.env.lower():
                action = np.random.choice([0,1],p=[0.95,0.05])
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
            #try:
            # update the trace
                # update the trace
            v_net.update_trace(state, beta)
            # update state values
            td_error = target - v_tilde
            v_net.online_update(td_error)

            # update beta values
            b_net.update_logits([(state, beta, target, v_tilde, v_tilde_prev, v)])
            error_list.append((np.abs(v_net.values - tmp_opt).mean(),np.abs(v_tilde - tmp_opt[state])))

            v_tilde_prev = (v_tilde - reward)/args.gamma
            #error_list.append(np.abs(v_tilde - tmp_opt[state]))

            if state in aliaser.aliased_indices:
                error_aliased = np.abs(v_tilde - tmp_opt[state])
                error_list_aliased.append(error_aliased)
                #experiment.log_metrics({"Error": error_aliased}, step=step)

            if step % args.print_every == 0 and args.rep == 1:
                error = np.array(error_list).mean(axis=0)

                error_list = []
                #experiment.log_metrics({"Error values": error[0],"Error V_tilde":error[1],"Learning rate":args.lr}, step=step)
                error_list = []
            if step % args.print_image == 0 and args.rep == 1:
                #plt.fill_between(np.arange(9), b_net.stable_sigmoid(b_net.b_logits),step="pre", alpha=0.4)
                #plt.step(np.arange(9),b_net.stable_sigmoid(b_net.b_logits))
                #experiment.log_figure(figure_name="Beta values")
                #plt.clf()
                #plt.fill_between(np.arange(9), v_net.values,step="pre", alpha=0.4)
                #plt.step(np.arange(9),b_net.stable_sigmoid(b_net.b_logits))
                #experiment.log_figure(figure_name="Values")

                plt.clf()
            state = next_state
        error_list_rep.append(error_list)
        error_list = []
        value_list.append(v_net.values)
    #print(np.array(value_list).mean(axis=0))
    error = np.array(error_list_rep).mean()*(-1)
    error_aliased = np.array(error_list_aliased).mean()*(-1)

    if error < -1  or math.isnan(error) or math.isinf(error):
        error = -1
    return error

#if args.hyper_comet:
for i in range(len(dic_param)):
    print("Step ",str(i+1) , "out of ", str(len(dic_param)))
    #suggestion = optimizer.get_suggestion()
    #experiment = Experiment("HFFoR5WtTjoHuBGq6lYaZhG0c",project_name = "ring-recurrent",workspace="pierthodo")
    experiment = ""
    score = fit(args,dic_param[i],experiment)
    dic_param[i]["score"] = score

        #suggestion.report_score("accuracy", score)
#else:
#    experiment = Experiment("HFFoR5WtTjoHuBGq6lYaZhG0c", project_name="ring-recurrent", workspace="pierthodo")
#    score = fit(args,"",experiment)


pprint(sorted(dic_param, key = lambda i: i['score'],reverse=True))
