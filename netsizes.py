#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Code which prints out performances of several different networks
"""


import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import trajectory_analyzer as ta
import trajectory_neuroevolution as tn

np.random.seed(0)


def get_xsw(net, steps, beta_start, beta_end, k_start, k_end, ss=0.001, dt=1):
    beta_max = max(beta_start, beta_end)
    k_max = max(k_start, k_end)
    beta = beta_start
    k = k_start
    beta_traj = [beta_start]
    k_traj = [k_start]
    times = [0]
    for j in range(0, steps): #let the neural net move
        inp = np.array([beta / beta_max, k / k_max]) #normalize beta and k
        out = net.run_network(inp)
        beta += out[0] * ss * beta_max
        k += out[1] * ss * k_max
        beta_traj.append(beta)
        k_traj.append(k)
        times.append(times[-1] + dt)
    dist_to_end = ((beta - beta_end))**2 + ((k - k_end))**2
    if dist_to_end > 0.1:
        beta_traj, k_traj, times = tn.go_to_target(beta_traj, k_traj, times, beta_end, k_end, dt, steps=10)
        score = ta.excess_work(beta_traj, k_traj, times)
    else:
        beta_traj, k_traj, times = tn.go_to_target(beta_traj, k_traj, times, beta_end, k_end, dt, steps=1)
        score = ta.excess_work(beta_traj, k_traj, times)
    return score


if __name__ == "__main__":
    generations = 32
    pbar = tqdm(total=generations)
    
    gen_nets = 100 #number of networks in a generation
    
    gencolors = {1:'red',2:'orange',4:'yellow',8:'green',16:'blue',32:'purple',64:'brown'}
    
    
    for h in [1,4,16,64,256,1024,4096,16384]:
        gen = tn.make_random_nets((h,), gensize=gen_nets)
        pbar = tqdm(total=generations)
        bestscores = []
        for i in range(0, generations):
            #play the game
            scores = [tn.play_game(gen[j], 100, 4, 10, 60, 15) for j in range(0, len(gen))]
            
            #sort the networks from best to worst
            combined = [(scores[k], gen[k]) for k in range(0, len(scores))]
            combined.sort(key=tn.f1, reverse = True)
            
            #Plot the best 5 performers of the generation
            if i == 0 or i==1 or i==3 or i==7 or i==15 or i==31 or i==63 or i==127 or i == 255:
                lbl = 'gen ' + str(i+1)
                tn.play_game(combined[0][1], 100, 4, 10, 60, 15, showplot=True, plotcolor=gencolors[i+1], plotlabel=lbl)
            bestscores.append(get_xsw(combined[0][1], 100, 4, 10, 60, 15))
            
            gen = tn.cull_the_weak(gen, scores)
            pbar.update()
        highscore = min(bestscores)
        plt.title(str(h) + ' Hidden Neurons, \n xs Work = ' + str(highscore))
        plt.xlabel('β')
        plt.ylabel('k')
        plt.legend()
        plt.savefig('hneurons' + str(h) + '.png')
        plt.show()