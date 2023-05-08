#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  9 16:37:34 2023

@author: joe
"""

import numpy as np
import copy
import matplotlib.pyplot as plt
from tqdm import tqdm
import trajectory_analyzer as ta

np.random.seed(0)



class Network():
    
    def __init__(self, w, b):
        self.w = w
        self.b = b
    
    #Runs the neural network
    def run_network(self, inp): #inp = numpy array of length 2, T and V
        #layer0 is the two inputs, or the T and V
        layer1 = 0.5*(np.tanh(np.matmul(self.w[0], inp)) + self.b[0])
        layer2 = np.matmul(self.w[1], layer1)
        return layer2
    
# Initializes a population of neural networks
#h is an array of the sizes of the hidden layers
def make_random_nets(h, gensize, inp=2, out=2):
    generation = []
    for l in range(0, gensize):
        w = []
        b = []
        w.append(np.random.randn(h[0], inp))
        b.append(np.random.randn(h[0]))
        for k in range(0, len(h)-1):
            w.append(np.random.randn(h[k+1],h[k]))
            b.append(np.random.randn(h[k+1]))
        w.append(np.random.randn(out,h[-1]))
        b.append(np.random.randn(out))
        generation.append(Network(w,b))
        #print(generation[-1].w[0][0,0])
    return generation

def plot_traj(traj_beta, traj_k, beta_start, beta_end, k_start, k_end, points=11, plotcolor='blue',plotlabel='none'):
    plt.plot([beta_start,beta_end], [k_start, k_end], color='black')
    plt.plot(traj_beta, traj_k, color=plotcolor,label=plotlabel)
    #plt.scatter(traj_beta[-points:], traj_k[-points:])

def vector_field(net, beta_min, beta_max, k_min, k_max, res):
    x = np.linspace(beta_min, beta_max, res)
    y = np.linspace(k_min, k_max, res)
    xx, yy = np.meshgrid(x, y)
    uu, vv = np.zeros_like(xx), np.zeros_like(xx)
    for i in range(res):
        for j in range(res):
            u, v = net.run_network(np.array([xx[i][j]/beta_max, yy[i][j]/k_max]))
            uu[i][j], vv[i][j] = u*0.001, v*0.001
    plt.quiver(xx, yy, uu, vv)

def go_to_target(beta_traj, k_traj, times, target_beta, target_k, dt, steps=10):
    beta_traj = beta_traj[:-steps]
    k_traj = k_traj[:-steps]
    times = times[:-steps]
    start_beta = beta_traj[-1]
    start_k = k_traj[-1]
    beta_left = target_beta - beta_traj[-1]
    k_left = target_k - k_traj[-1]
    for j in range(steps):
        k_traj.append(start_k + k_left * (j+1)/steps)
        beta_traj.append(start_beta + beta_left * (j+1)/steps)
        times.append(times[-1] + dt)
    return (beta_traj, k_traj, times)

#K is the total number of steps
def play_game(net, steps, beta_start, beta_end, k_start, k_end, ss=0.001, dt=1, showplot=False, showvec=False, plotcolor='blue', plotlabel='none'):
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
    if dist_to_end > 10:
        score = dist_to_end*10
    elif dist_to_end > 0.1:
        beta_traj, k_traj, times = go_to_target(beta_traj, k_traj, times, beta_end, k_end, dt, steps=10)
        score = ta.excess_work(beta_traj, k_traj, times)
    else:
        beta_traj, k_traj, times = go_to_target(beta_traj, k_traj, times, beta_end, k_end, dt, steps=1)
        score = ta.excess_work(beta_traj, k_traj, times)
    if showplot:
        plot_traj(beta_traj, k_traj,  beta_start, beta_end, k_start, k_end, points=1, plotcolor=plotcolor, plotlabel=plotlabel)
    if showvec:
        vector_field(net, min(beta_start, beta_end),beta_max,min(k_start, k_end),k_max,20)
    return -score


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
        beta_traj, k_traj, times = go_to_target(beta_traj, k_traj, times, beta_end, k_end, dt, steps=10)
        score = ta.excess_work(beta_traj, k_traj, times)
    else:
        beta_traj, k_traj, times = go_to_target(beta_traj, k_traj, times, beta_end, k_end, dt, steps=1)
        score = ta.excess_work(beta_traj, k_traj, times)
    return score

def mutate(net, var = 1, step_size = 0.05):
    net2 = copy.deepcopy(net)
    for element in net2.w:
        element += np.random.normal(loc=0.0, scale=var, size=np.shape(element))*step_size
    for element in net2.b:
        element += np.random.normal(loc=0.0, scale=var, size=np.shape(element))*step_size
    return net2

def f1(obj):
    return obj[0]

def cull_the_weak(gen, scores): # take out all players that are below median
    combined = [(scores[k], gen[k]) for k in range(0, len(scores))]
    combined.sort(key=f1, reverse = True)
    nextgen = []
    for k in range(0, 25):
        nextgen.append(combined[k][1])
        nextgen.append(mutate(combined[k][1]))
        nextgen.append(mutate(combined[k][1]))
        nextgen.append(mutate(combined[k][1]))
    return nextgen



if __name__ == '__main__':
    gen_nets = 100 #number of networks in a generation
    gen = make_random_nets((1024,), gensize=gen_nets)
    
    generations = 128
    
    medians = []
    maxs = []
    x = range(0, generations)
    
    pbar = tqdm(total=generations)
    for g in range(0, generations):
        scores = [play_game(gen[j], 100, 4, 10, 60, 15) for j in range(0, len(gen))]
        gen = cull_the_weak(gen, scores)
        medians.append(np.median(scores))
        maxs.append(max(np.max(scores), -1))
        if g == 0 or g==1 or g==3 or g==7 or g==15 or g==31 or g==63 or g==127 or g==255:
            print(play_game(gen[0], 100, 4, 10, 60, 15, showplot=True, showvec=True,plotcolor='blue', plotlabel=str(g)))
            plt.xlabel('β')
            plt.ylabel('k')
            best = get_xsw(gen[0], 100, 4, 10, 60, 15)
            plt.title('Optimal Trajectory in Generation ' + str(g+1) + '\n xs Work = ' + str(best))
            plt.savefig('policy' + str(g+1) + '.png')
            plt.show()
        if g == 127:
            print(play_game(gen[0], 100, 4, 10, 60, 15, showplot=True, showvec=False))
            plt.xlabel('β')
            plt.ylabel('k')
            plt.title('Optimal Trajectory')
            plt.savefig('traj' + str(g) + '.png')
            plt.show()
            
        pbar.update(1)
        
    plt.plot(x, maxs)
    plt.show()








