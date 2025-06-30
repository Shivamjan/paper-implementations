from __future__ import print_function

import torch
import torch.nn as nn

import matplotlib.pyplot as plt

import random

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
