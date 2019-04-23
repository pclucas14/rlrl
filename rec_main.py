import sys 
import torch
import argparse
import torch.nn as nn
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
parser.add_argument('--alias_percentage',  type=float, default=0)
parser.add_argument('--lambda', type=float, default=0.5) # -1 for fully online
parser.add_argument('--online', type=int, default=0)
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
v_net   = V_net(args, env, aliaser)

model      = nn.GRU(env.n, 50)
projection = nn.Linear(50, 1)
state_idx  = torch.zeros(env.n).view(1, 1, -1)
gru_hidden = None

opt = torch.optim.Adam([{'params':model.parameters(), 'lr':1e-2}, {'params':projection.parameters(), 'lr':1e-2}])

for episode in range(args.n_episodes):
    done = False
    state = env.reset()
    t = 0    

    # buffer for beta updates
    memory_value = []

    model.eval()
    while not done:
        with torch.no_grad():

            # we are evaluating random policies. we pick a random action
            action = env.action_space.sample()
            next_state, reward, done, _ = env.step(action)

            ''' ALIASING HAPPENS HERE '''
            if not done:
                next_state = aliaser(next_state)

            # one-hot encoding for the state
            state_idx.fill_(0)
            state_idx[:, :, next_state] = 1

            # build target
            # target = reward
            if not done: 
                next_state_hid, gru_hidden = model(state_idx)
                next_state_value = projection(next_state_hid).item()
                # target += args.gamma * state_value

            # update the trace
            if args.online: 
                raise ValueError("not supported for now")
            else:
                # store all required_values for value update
                memory_value += [(state, next_state, reward, next_state_value)]

            state = next_state 
            t += 1
            
    if not args.online:

        # get lambda or MC targets
        targets = v_net.get_targets(memory_value, for_gru=True)
        states  = [x[0] for x in memory_value]
        
        targets = torch.Tensor(targets)
        states  = torch.Tensor(states).long()
        states_one_hot = torch.zeros(states.size(0), env.n)
        states_one_hot.scatter_(1, states.unsqueeze(1), 1)

        output, hid = model(states_one_hot.unsqueeze(1)) # batch is 2nd axis
        state_value_pred = projection(output.squeeze(1)).squeeze(-1)

        opt.zero_grad()
        loss = (targets - state_value_pred) ** 2
        loss.mean().backward()
        opt.step()

    if (episode + 1) % args.print_every == 0:
        print('states', states)
        print('pred. value', state_value_pred)
        print('targets', targets)
        # print values
