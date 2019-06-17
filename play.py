from model import DQN
from pong import Pong
from breakout import Breakout
import torch
import torch.autograd as autograd
from utils import sample_action
import time
import numpy as np

the_model = torch.load('./tmp/model.pth')

method = 'DQN' # DQN or BDQN
use_eGreedy = 0 # 1: true; 0: false
game = 'pong' # pong or breakout

if method == 'DQN':
    num_heads = 1
elif method == 'BDQN':
    num_heads = 10
    
if use_eGreedy == 1:
    epsilon = 1.0
elif use_eGreedy == 0:
    epsilon = 0.0

if game == 'pong':
    VALID_ACTION = [0, 2, 5] # 0:no-op 2:up 5:down
    atari = Pong()
elif game == 'breakout':
    VALID_ACTION = [0, 1, 2, 3] # 0:no-op 1:fire 2:left 3:right
    atari = Breakout()

num_actions = len(VALID_ACTION)

dqn_heads = []
act_index = np.zeros(num_heads,dtype=int)
epsilon_eval = 0.05
done = False

for i in range(num_heads):
    the_model = torch.load('./tmp/model_epoch_200/model'+str(i)+'.pth')
    dqn_heads.append(DQN(num_actions))
    dqn_heads[i].load_state_dict(the_model)

#dqn = DQN()
#dqn.load_state_dict(the_model)

with torch.no_grad():
    var_phi = autograd.Variable(torch.Tensor(1, 4, 84, 84))

while(not done):

    for i in range(num_heads):
        act_index[i] = sample_action(atari, dqn_heads[i], var_phi, epsilon = epsilon_eval, num_actions)
        
    act_index_vote = np.bincount(act_index).argmax()

    phi_next, r, done = atari.step(VALID_ACTION[act_index_vote])
    atari.display()
    time.sleep(0.01)
