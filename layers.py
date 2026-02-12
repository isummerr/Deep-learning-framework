import random
import math
from tensor import Tensor

class Layer:
    def parameters(self):
        return self.params

class Conv2d(Layer):
    def __init__(self, in_channels, out_channels, kernel_size):
        self.in_c, self.out_c, self.k = in_channels, out_channels, kernel_size
        # Xavier Initialization for stability
        limit = math.sqrt(6 / (in_channels + out_channels))
        self.weight = Tensor([[[[random.uniform(-limit, limit) for _ in range(self.k)] 
                         for _ in range(self.k)] for _ in range(self.in_c)] 
                        for _ in range(self.out_c)], requires_grad=True)
        self.bias = Tensor([0.0 for _ in range(out_channels)], requires_grad=True)
        self.params = [self.weight, self.bias]
        self.last_macs = 0

    def forward(self, x):
        b_sz, in_h, in_w = len(x), len(x[0][0]), len(x[0][0][0])
        out_h, out_w = in_h - self.k + 1, in_w - self.k + 1
        res = [[[[0.0 for _ in range(out_w)] for _ in range(out_h)] 
                   for _ in range(self.out_c)] for _ in range(b_sz)]
        for b in range(b_sz):
            for oc in range(self.out_c):
                for ic in range(self.in_c):
                    for i in range(out_h):
                        for j in range(out_w):
                            for ki in range(self.k):
                                for kj in range(self.k):
                                    res[b][oc][i][j] += x[b][ic][i+ki][j+kj] * self.weight.data[oc][ic][ki][kj]
                for i in range(out_h):
                    for j in range(out_w): 
                        res[b][oc][i][j] += self.bias.data[oc]
        # Mandatory Reporting Metric calculation
        self.last_macs = b_sz * out_h * out_w * self.out_c * (self.k * self.k * self.in_c)
        return Tensor(res, (self.weight, self.bias), 'conv')

class ReLU:
    def forward(self, x):
        # Using a tiny slope (0.01) for negative values to keep gradients flowing
        # This is a 'Leaky ReLU' - much better for deep learning assignments
        return [[[[max(0.01 * v, v) for v in row] for row in ch] for ch in b] for b in x]

class MaxPool2d:
    def __init__(self, kernel_size=2): # CHANGED: Accepts kernel_size to match train.py
        self.k = kernel_size
        
    def forward(self, x):
        b, c, h, w = len(x), len(x[0]), len(x[0][0]), len(x[0][0][0])
        oh, ow = h // self.k, w // self.k
        res = [[[[0.0 for _ in range(ow)] for _ in range(oh)] for _ in range(c)] for _ in range(b)]
        for b_idx in range(b):
            for c_idx in range(c):
                for i in range(oh):
                    for j in range(ow):
                        window = [x[b_idx][c_idx][i*self.k+ki][j*self.k+kj] 
                                  for ki in range(self.k) for kj in range(self.k)]
                        res[b_idx][c_idx][i][j] = max(window)
        return res

class Flatten:
    def forward(self, x):
        return [[v for ch in b for row in ch for v in row] for b in x]

class Linear(Layer):
    def __init__(self, in_features, out_features): # Updated names here
        self.in_f = in_features
        self.out_f = out_features
        limit = math.sqrt(6 / (in_features + out_features))
        self.weight = Tensor([[random.uniform(-limit, limit) for _ in range(in_features)] 
                        for _ in range(out_features)], requires_grad=True)
        self.bias = Tensor([0.0 for _ in range(out_features)], requires_grad=True)
        self.params = [self.weight, self.bias]
        self.last_macs = 0

    def forward(self, x):
        b_sz = len(x)
        # Standard matrix multiplication: y = xW^T + b
        # len(x[0]) is the number of input features
        res = [[sum(x[b][j] * self.weight.data[i][j] for j in range(len(x[0]))) + self.bias.data[i] 
                for i in range(self.out_f)] for b in range(b_sz)]
        self.last_macs = b_sz * self.out_f * self.in_f
        return Tensor(res, (self.weight, self.bias), 'linear')