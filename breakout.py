
import sys
sys.path.append('/u/training/tra413/.local/lib/python3.6/site-packages/')
sys.path.append('/home/lihan/.local/lib/python2.7/site-packages')
sys.path.append('/opt/packages/python/2_7_11_gcc/lib/python2.7/site-packages')
sys.path.append('/home/lihan/RL')
sys.path.append('/home/lihan/RL/gym')
sys.path.append('/home/lihan/RL/gym/gym')
import gym
import numpy as np
import cv2

class Breakout(object):

    def __init__(self):

        self.env = gym.make('BreakoutDeterministic-v4')
        self.current_phi = None
        self.reset()
        self.lives = 6

    def step(self, action):
		
        obs, r, done, info = self.env.step(action)

        if( self.lives == 6 ):
            obs, r, done, info = self.env.step(1) # fire
            self.lives -= 1

        if( info['ale.lives'] == 4 ):
            done = True

#        obs1 = obs.transpose(2,0,1)
        obs = self._rbg2gray(obs)
        phi_next = self._phi(obs)

        phi_phi = np.vstack([self.current_phi, obs[np.newaxis]])
        self.current_phi = phi_next

        return phi_phi, r, done

    def reset(self):
        x = self.env.reset()
        x = self._rbg2gray(x)
        phi = np.stack([x, x, x, x])
        self.current_phi = phi
        self.lives = 6

        return phi

    def _rbg2gray(self, x):
        x = x.astype('float32')
        x = cv2.cvtColor(x, cv2.COLOR_RGB2GRAY)
        x = cv2.resize(x, (84, 84))/127.5 - 1.

        return x

    def _phi(self, x):

        new_phi = np.zeros((4, 84, 84), dtype=np.float32)
        new_phi[:3] = self.current_phi[1:]
        new_phi[-1] = x
		
        return new_phi

    def display(self):
        self.env.render()
