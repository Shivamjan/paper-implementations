from __future__ import print_function

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import matplotlib.pyplot as plt

import os,sys,random,time
import argparse

from main import *


gamma = [0, 0.5, 1, 2, 3, 4, 5]
errors=[]

for g in gamma:
    max_e = 0
    for i in range(1000):
        x = torch.rand(12800, 2) * random.randint(1, 10)
        y = torch.rand(12800).ge(0.1).long()

        ce = nn.CrossEntropyLoss()(x,y).item()
        focal_loss = FocalLoss(gamma=g)(x, y).item()
        max_e = max(max_e, abs(ce - focal_loss))
    errors.append(max_e)
plt.figure(figsize=(10,10))
plt.plot(gamma, errors, marker ='o')
plt.xlabel('gamma')
plt.ylabel('Max error from CE')
plt.grid(True)
plt.show()
#
# start_time = time.time()
# max_e = 0
# for i in range(1000):
#     x = torch.rand(12800, 2)*random.randint(1,10)
#     y = torch.rand(12800).ge(0.1).long()
#
#     output0 = FocalLoss(gamma=2)(x, y)
#     output1 = nn.CrossEntropyLoss()(x, y)
#
#
#     a = output0.item()
#     b = output1.item()
#
#     if abs(a-b)>max_e: max_e = abs(a-b)
# print('time:', time.time()-start_time, 'max_error:', max_e)