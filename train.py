import sys
#sys.path.append('/mnt/bwpy/single/usr/lib/python3.5/site-packages/gym')
import os

from model import DQN
import torch
import torch.autograd as autograd
import torch.optim as optim

import numpy as np
import random

from pong import Pong
from breakout import Breakout
from memory import MemoryReplay

import time
from utils import (sample_action, save_statistic)


method = 'DQN' # DQN or BDQN
use_eGreedy = 1 # 1: true; 0: false
game = 'pong' # pong or breakout

if method == 'DQN':
    num_heads = 1
    p = 1.0
elif method == 'BDQN':
    num_heads = 5
    p = 0.5 # mask distribution parameter
    
if use_eGreedy == 1:
    epsilon = 1.0
elif use_eGreedy == 0:
    epsilon = 0.0

if game == 'pong':
    VALID_ACTION = [0, 2, 5] # 0:no-op 2:up 5:down
    atari = Pong()
    avg_score = -21.0
    best_score = -21.0
    avg_score_eval = -21.0
    best_score_eval = -21.0
elif game == 'breakout':
    VALID_ACTION = [0, 1, 2, 3] # 0:no-op 1:fire 2:left 3:right
    atari = Breakout()
    avg_score = 0.0
    best_score = 0.0
    avg_score_eval = 0.0
    best_score_eval = 0.0

num_actions = len(VALID_ACTION)

epsilon_eval = 0.05
GAMMA = 0.99
update_step = 1000
memory_size = 100000
max_epoch = 1000
batch_size = 32
LR = 0.0025

save_path = './results_' + game + '_' + method + '_nHeads_' + str(num_heads) + '_eGreedy_' + str(use_eGreedy)

if not os.path.exists(save_path):
    os.makedirs(save_path)

# Variables
with torch.no_grad():
    var_phi = autograd.Variable(torch.Tensor(1, 4, 84, 84)).cuda()

# For training
var_batch_phi = autograd.Variable(torch.Tensor(batch_size, 4, 84, 84)).cuda()
var_batch_a = autograd.Variable(torch.LongTensor(batch_size, 1), requires_grad=False).cuda()
var_batch_r = autograd.Variable(torch.Tensor(batch_size, 1)).cuda()
var_batch_phi_next = autograd.Variable(torch.Tensor(batch_size, 4, 84, 84)).cuda()
var_batch_r_mask = autograd.Variable(torch.Tensor(batch_size, 1), requires_grad=False).cuda()

MP = MemoryReplay(max_size=memory_size, bs=batch_size, im_size=84, stack=4, num_heads = num_heads, p = p)
dqn_heads = []
target_dqn_heads = []
optimz_heads = []
for i in range(num_heads):
    dqn_heads.append(DQN(num_actions))
    target_dqn_heads.append(DQN(num_actions))
    target_dqn_heads[i].load_state_dict(dqn_heads[i].state_dict())

    dqn_heads[i].cuda()
    target_dqn_heads[i].cuda()

    optimz = optim.RMSprop(dqn_heads[i].parameters(),  lr=LR, alpha=0.95, eps=1e-02, momentum=0.0)
    optimz_heads.append(optimz)

for i in range(memory_size):
    phi = atari.current_phi
    act_index = random.randrange(num_actions)
    phi_next, r, done = atari.step(VALID_ACTION[act_index])
    
#    atari.display()
    MP.put((phi_next, act_index, r, done))

    if done:
        atari.reset()

print("================\n" 
	  "Start training!!\n"
	  "================")
atari.reset()

epoch = 0
step_count = 0
update_count = np.zeros(num_heads)
score = 0.0

t = time.time()
t0 = time.time()

SCORE = []
SCORE_EVAL = []
QVALUE = []
QVALUE_MEAN = []
QVALUE_STD = []
STEP_COUNT = []

while(epoch < max_epoch): 
    k_head = np.random.randint(num_heads, size=1)[0]
    
    while(not done):
        
        act_index = sample_action(atari, dqn_heads[k_head], var_phi, epsilon, num_actions)
        
        if use_eGreedy == 1:
            epsilon = (epsilon - 1e-6) if epsilon > 0.1 else  0.1
        elif use_eGreedy == 0:
            epsilon = 0.0
            
        phi_next, r, done = atari.step(VALID_ACTION[act_index])
#        atari.display()
        MP.put((phi_next, act_index, r, done))
        r = np.clip(r, -1, 1)
        score += r
        
        for i in range(num_heads):
            optimz_heads[i].zero_grad()
            
            # batch sample from memory to train
            batch_phi, batch_a, batch_r, batch_phi_next, batch_done = MP.batch(i)
            var_batch_phi_next.data.copy_(torch.from_numpy(batch_phi_next))
            batch_target_q, _ = target_dqn_heads[i](var_batch_phi_next).max(dim=1)      
    
            mask_index = np.ones((batch_size, 1))
            mask_index[batch_done] = 0.0
            var_batch_r_mask.data.copy_(torch.from_numpy(mask_index))
    		
            var_batch_r.data.copy_(torch.from_numpy(batch_r).view(-1,1))
            
            batch_target_q = batch_target_q.reshape(var_batch_r.shape)
    
            y = var_batch_r + batch_target_q.view(-1,1).mul(GAMMA).mul(var_batch_r_mask)
            y = y.detach()
    
            var_batch_phi.data.copy_(torch.from_numpy(batch_phi))
            batch_q = dqn_heads[i](var_batch_phi)
    
            var_batch_a.data.copy_(torch.from_numpy(batch_a).long().view(-1, 1))
            batch_q = batch_q.gather(1, var_batch_a)
    		
            loss = y.sub(batch_q).pow(2).mean()
            loss.backward()
            optimz_heads[i].step()
    
            update_count[i] += 1
            step_count += 1
    
            if update_count[i] == update_step:
                    target_dqn_heads[i].load_state_dict(dqn_heads[i].state_dict())
                    update_count[i] = 0
    
            QVALUE.append(batch_q.data.cpu().numpy().mean())

    SCORE.append(score)
    QVALUE_MEAN.append(np.mean(QVALUE))
    QVALUE_STD.append(np.std(QVALUE))
    QVALUE = []

    save_statistic('Score', SCORE, save_path=save_path)
    save_statistic('Average Action Value', QVALUE_MEAN, QVALUE_STD, save_path)
    
    np.save(save_path+'/SCORE.npy', SCORE)
    np.save(save_path+'/QVALUE_MEAN.npy', QVALUE_MEAN)
    np.save(save_path+'/QVALUE_STD.npy', QVALUE_STD)
    
    STEP_COUNT.append(step_count)
    save_statistic('Step', STEP_COUNT, save_path=save_path)
    np.save(save_path+'/STEP.npy', STEP_COUNT)

    atari.reset()
    done = False
    epoch += 1
    avg_score = 0.9*avg_score + 0.1*score
    score = 0.0
    
    # ensssemble policy
    while(not done):
        
        action = np.zeros(num_actions)
        for i in range(num_heads):
            act_index = sample_action(atari, dqn_heads[i], var_phi, epsilon_eval, num_actions)
            action[act_index] +=1
        act_index = list(action).index(max(action))
        
        phi_next, r, done = atari.step(VALID_ACTION[act_index])
#        atari.display()
        r = np.clip(r, -1, 1)
        score += r
    
    SCORE_EVAL.append(score)
    save_statistic('Score_eval', SCORE_EVAL, save_path=save_path)
    np.save(save_path+'/SCORE_EVAL.npy', SCORE_EVAL)
    avg_score_eval = 0.9*avg_score_eval + 0.1*score
    score = 0.0
    atari.reset()
    done = False
    time_elapse = time.time() - t
    t_elapse = (time.time() - t0)

    print("Epoch %4d" % epoch, " Epsilon %.6f" % (epsilon), " Avg.Score %.2f" % (avg_score_eval), " Time Elapse %.2f min" % float(t_elapse/60.0))
    sys.stdout.flush()

    if avg_score_eval >= best_score_eval and time_elapse > 300:
        for i in range(num_heads):
            torch.save(dqn_heads[i].state_dict(), save_path+'/model'+str(i)+'.pth')
        print('Models have been saved.')
        best_score_eval = avg_score_eval
        t = time.time()
