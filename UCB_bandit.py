#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb  6 23:05:33 2021

@author: thomas
"""

import numpy as np
import matplotlib.pyplot as plt

bandit_prob = [0.2,0.5,0.75]
num_trials  = 100000

class UCB_Bandit:
    def __init__(self,p):
        self.p     = p   # true mean
        self.p_est = 15. # estimated mean from simulation
        self.N     = 0.
        
    def pull(self):
        # draw a 1 with a probability p
        return np.random.random() < self.p 
        
    def samp_mean(self,x):
        return self.p_est + (x - self.p_est) / self.N
    
    def update(self,x):
        # update the 
        self.N    += 1.
        self.p_est = self.samp_mean(x)


def ucb(mean,n,nj):
    return mean + np.sqrt(2*np.log(n) / nj)


def run_experiment():
    bandits     = [UCB_Bandit(p) for p in bandit_prob]
    rewards     = np.empty(num_trials)
    total_plays = 0
    
    # initialize with playing all bandits once
    for j in range(len(bandits)):
        # play the bandit
        x = bandits[j].pull()
        # update total plays
        total_plays += 1
        # update the estimated mean
        bandits[j].update(x)
        
        
    for i in range(num_trials):
        j = np.argmax([ucb(b.p_est,total_plays,b.N) for b in bandits])
        x = bandits[j].pull()
        total_plays += 1
        bandits[j].update(x)
        
        rewards[i] = x
        
    cumulative_average = np.cumsum( rewards )/( np.arange(num_trials) + 1 )
    
    #return cumulative_average
    
    # print the mean estimate for each bandit
    for b in bandits:
        print("mean estimate:",b.p_est)
    
    # print total reward
    print("total reward:",rewards.sum())

    plt.plot(cumulative_average)
    plt.plot(np.ones(num_trials)*np.max(bandit_prob))
    plt.xscale('log')
    plt.legend(['True rate','Optimal rate'])
    plt.show()

if __name__ == '__main__':
    run_experiment()
        
        
        
        
        
    
    