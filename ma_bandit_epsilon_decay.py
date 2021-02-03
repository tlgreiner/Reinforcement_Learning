#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  2 12:44:35 2021

@author: thomas

Initial script from Lazy programmers course 
in Reinforcement learning.
Modified further for testing purpose.
"""

import matplotlib.pyplot as plt
import numpy as np

class ArmedBandit:
    def __init__(self,m,eps):
        # m is the true mean probability of the bandit 
        self.m          = m 
        self.m_estimate = 0.0
        self.N          = 1.0
        self.eps        = eps
        self.eps0       = eps
        
    def pull(self):
        # draw with gaussian distribution
        return np.random.randn() +self.m
    
    def samp_mean(self,x):
        return self.m_estimate + (x - self.m_estimate) / self.N
    
    def update(self,x):
        # update the 
        self.N         += 1.
        self.m_estimate = self.samp_mean(x)
        
    def update_eps(self):
        t     = self.N
        self.eps = 1/np.log( t + 1 )
    
       
def run_experiment(M,eps0,N):
    # create the bandits with the true mean probabilities
    bandits = [ArmedBandit(m,eps0) for m in M]
    
    # the true means
    true_means = np.array(M)
    true_best  = np.argmax(true_means) 
    count_suboptimal = 0
    
    # store epsilon
    eps = eps0
    
    data = np.empty(N)
    j    = 0
    for i in range(N):
        
        p = np.random.random()
        # epsilon greedy with epsilon decay
        if p < eps:
            j = np.random.randint(len(bandits))
        else:
            # choose the bandit with highest estimate
            j = np.argmax([b.m_estimate for b in bandits])
        # pull the arm for the bandit with the largest sample
        x = bandits[j].pull()
        # update the distribution for the bandit we just pulled
        bandits[j].update(x)   
        bandits[j].update_eps()
        eps = bandits[j].eps
        
        if j != true_best:
            count_suboptimal += 1
        # store rewards in data
        data[i] = x  
            
    cumulative_average = np.cumsum(data) / (np.arange(N) + 1)
        
    plt.plot(cumulative_average)
    [plt.plot(np.ones(N)*M[i]) for i in range(len(M))]
    plt.xscale('log')
    plt.show()
    
    for b in bandits:
        print(b.m_estimate)
        
    print( "percent suboptimal for epsilon = %s:" % eps0, float(count_suboptimal)/N )
    
    return cumulative_average



if __name__ == '__main__':
    M    = [ 1.5 , 2.5 , 3.5 ]
    N    = 200000
    eps0 = [ 0.1 , 0.05 , 0.01]
    
    exp0 = run_experiment(M,eps0[0],N)
    exp1 = run_experiment(M,eps0[1],N)
    exp2 = run_experiment(M,eps0[2],N)
    
    # log scale plot
    plt.plot(exp0, label='eps=0.1')
    plt.plot(exp1, label='eps=0.05')
    plt.plot(exp2, label='eps=0.01')
    plt.legend()
    plt.xscale('log')
    plt.show()
    
    # linear scale plot
    plt.plot(exp0, label='eps=0.1')
    plt.plot(exp1, label='eps=0.05')
    plt.plot(exp2, label='eps=0.01')
    plt.legend()
    plt.show()
    
    
    
    
    
    
    
    
    
    
