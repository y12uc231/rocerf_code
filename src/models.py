import torch
import torch.nn as nn
import torch.nn.functional as F

class LR_Model(nn.Module):
    def __init__(self, num_input):
        super().__init__()
        self.linear = nn.Linear(num_input, 1)

    def forward(self, x):
        return self.linear(x)
    

    
class NN3_ModelSP(nn.Module):
    def __init__(self, num_input, beta = 1):
        super().__init__()
        self.linear1 = nn.Linear(num_input, 2*num_input)
        self.linear2 = nn.Linear(2*num_input, 2*num_input)
        self.linear3 = nn.Linear(2*num_input, 1)
        self.beta = beta
        self.softplus = nn.Softplus(beta=beta)
    
    def c_softplus(self, x):
        return self.softplus(x) - torch.log(torch.tensor([2])) / self.beta

    def forward(self, x):
        return self.linear3(self.c_softplus(self.linear2(self.c_softplus(self.linear1(x)))))