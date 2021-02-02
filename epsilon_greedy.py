#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb  1 21:52:20 2021

@author: thomas

Initial script from Lazy programmers course 
in Reinforcement learning.
Modified further for testing purpose.
"""

import matplotlib.pyplot as plt
import numpy as np

NUM_TRIALS = 10000
EPS         = 0.1
BANDIT_PROB = [0.2,0.5,0.75]

class Bandit:
    def __init__(self,p):
        # p is the probability of the bandit 
        self.p          = p 
        self.p_estimate = 0.
        self.N          = 1.
        
    def pull(self):
        # draw a 1 with a probability p. Boolean True/False
        return np.random.random() < self.p
    
    def samp_mean(self,x):
        return self.p_estimate + (x - self.p_estimate) / self.N
    
    def update(self,x):
        # update the 
        self.N         += 1.
        self.p_estimate = self.samp_mean(x)
        
def experiment():
    # create the bandits with the bandit probability of giving reward
    bandits = [Bandit(p) for p in BANDIT_PROB]
    
    rewards = np.zeros(NUM_TRIALS)
    
    num_times_explored = 0
    num_times_exploit  = 0
    num_optimal        = 0
    # select optimal bandit from the bandit with highest prob 
    optimal_j          = np.argmax([b.p for b in bandits])
    print("optimal j:",optimal_j)
    
    
    for i in range(NUM_TRIALS):
        
        # use epsilon greedy to select the next bandit
        if np.random.random() < EPS:
            
            num_times_explored += 1
            # choose a random bandit
            j = np.random.randint(len(bandits))
        else:
            num_times_exploit  += 1
            # choose the bandit with highest estimate
            j = np.argmax([b.p_estimate for b in bandits])
            
        if j == optimal_j:
            num_optimal += 1
            
            
        # pull the arm for the bandit with the largest sample
        x = bandits[j].pull()
    
        # update rewards log
        rewards[i] = x
    
        # update the distribution for the bandit we just pulled
        bandits[j].update(x)
        
    
    # print the mean estimate for each bandit
    for b in bandits:
        print("mean estimate:",b.p_estimate)
    
    # print total reward
    print("total reward:",rewards.sum())
    print("overall winrate:",rewards.sum() / NUM_TRIALS)
    print("number of times explored:",num_times_explored)
    print("number of times exploited:",num_times_exploit)
    print("number of times selected optimal bandit:",num_optimal)
    
    cumulative_rewards = np.cumsum(rewards)
    win_rates          = cumulative_rewards / (np.arange(NUM_TRIALS)+1)
    plt.plot( win_rates )
    plt.plot( np.ones(NUM_TRIALS)*np.max(BANDIT_PROB) )
    plt.legend(['True rate','Optimal rate'])
    plt.show()
    
if __name__ == '__main__':
   experiment()



    
    
    





      
        
        
        
        
        
        
        
        
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                