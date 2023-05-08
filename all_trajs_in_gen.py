#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Code which prints out performers at several generations
"""


import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import trajectory_analyzer as ta
import trajectory_neuroevolution as tn

np.random.seed(0)



generations = 256
pbar = tqdm(total=generations)

gen_nets = 100 #number of networks in a generation
gen = tn.make_random_nets((1024,), gensize=gen_nets)

for i in range(0, generations):
    #play the game
    scores = [tn.play_game(gen[j], 100, 4, 10, 60, 15) for j in range(0, len(gen))]
    
    #sort the networks from best to worst
    combined = [(scores[k], gen[k]) for k in range(0, len(scores))]
    combined.sort(key=tn.f1, reverse = True)
    
    #Plot the best 5 performers of the generation
    if i == 0 or i==1 or i==3 or i==7 or i==15 or i==31 or i==63 or i==127 or i == 255:
        tn.play_game(combined[0][1], 100, 4, 10, 60, 15, showplot=True, plotcolor='red', plotlabel='best')
        tn.play_game(combined[1][1], 100, 4, 10, 60, 15, showplot=True, plotcolor='orange', plotlabel='2nd best')
        tn.play_game(combined[2][1], 100, 4, 10, 60, 15, showplot=True, plotcolor='yellow', plotlabel='3rd best')
        tn.play_game(combined[3][1], 100, 4, 10, 60, 15, showplot=True, plotcolor='green', plotlabel='4th best')
        tn.play_game(combined[4][1], 100, 4, 10, 60, 15, showplot=True, plotcolor='blue', plotlabel='5th best')
        plt.title('Best performers from generation ' + str(i+1))
        plt.xlabel('β')
        plt.ylabel('k')
        plt.legend()
        plt.savefig('bestperformersgen' + str(i+1) + '.png')
        plt.show()
    
    #Plot the worst 5 performers of the generation
    if i == 0 or i==1 or i==3 or i==7 or i==15 or i==31 or i==63 or i==127 or i == 255:
        tn.play_game(combined[-1][1], 100, 4, 10, 60, 15, showplot=True, plotcolor='red', plotlabel='worst')
        tn.play_game(combined[-2][1], 100, 4, 10, 60, 15, showplot=True, plotcolor='orange', plotlabel='2nd worst')
        tn.play_game(combined[-3][1], 100, 4, 10, 60, 15, showplot=True, plotcolor='yellow', plotlabel='3rd worst')
        tn.play_game(combined[-4][1], 100, 4, 10, 60, 15, showplot=True, plotcolor='green', plotlabel='4th worst')
        tn.play_game(combined[-5][1], 100, 4, 10, 60, 15, showplot=True, plotcolor='blue', plotlabel='5th worst')
        plt.title('Worst performers from generation ' + str(i+1))
        plt.xlabel('β')
        plt.ylabel('k')
        plt.legend()
        plt.savefig('worstperformersgen' + str(i+1) + '.png')
        plt.show()
    
    gen = tn.cull_the_weak(gen, scores)
    pbar.update()






