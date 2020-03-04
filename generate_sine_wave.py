import numpy as np
import torch
import random
import os



np.random.seed(2)

T = 20
L = 1000
N = 100

x = np.empty((N, L), 'int64')
print (x.shape)
x[:] = np.array(range(L)) + np.random.randint(-4 * T, 4 * T, N).reshape(N, 1)
print (x.shape)
print (x)
data = np.sin(x / 1.0 / T).astype('float64')
print ("data: ", data)
print ("data.shape: ", data.shape)
torch.save(data, open('traindata.pt', 'wb'))
