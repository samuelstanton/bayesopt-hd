import math
import torch
from torch.autograd import Variable

#source: www.sfu.ca/~ssurjano/optimization

def branin(scaled_x):
    #Recommended parameter values for rescaled branin on unit rectangle
    a = 1/51.95; b=5.1/(4*math.pi**2)
    c=5/math.pi; r=6 
    s=10; t=1/(8*math.pi)
    
    if len(scaled_x.shape) == 1:
        scaled_x = scaled_x.view(1, 2)
    
    x = Variable(torch.zeros(scaled_x.shape[0], 2))
    x[:, 0] = 15 * scaled_x[:, 0] - 5
    x[:, 1] = 15 * scaled_x[:, 1]
    
    return -a*((x[:, 1] - b*x[:, 0]**2 + c*x[:, 0] - r)**2 + s*(1-t)*torch.cos(x[:, 0]) - 44.81)


def decaying_cos(x):
    n = x.shape[-1]
    if len(x.shape) == 1:
        x = x.view(1, n)
    val = Variable(torch.zeros(x.shape[0]))
    for i in range(n):
        val = val + -1/(x[:, i]+1)*torch.cos(2*math.pi*x[:, i])
    return 1/n * val


def rosenbrock(scaled_x):
    dim = scaled_x.shape[-1]
    if len(scaled_x.shape) == 1:
        scaled_x = scaled_x.view(1, dim)
    
    x = 15 * scaled_x - 5
    a = 1/(3.755*10**5)
    b = -3.827*10**5
    val = Variable(torch.zeros(x.shape[0]))
    
    for i in range(dim-1):
        val = val + 100*(x[:, i+1] - x[:, i]**2)**2 + (1 - x[:, i])**2
        
    return -a*(val + b)


def hartmann6d(x):
    #Global maximum 3.32237 @ (0.20169, 0.150011, 0.476874, 0.275332, 0.311652, 0.6573)
    n = x.shape[-1]
    if len(x.shape) == 1:
        x = x.view(1, n)
        
    alpha = torch.Tensor([1.0, 1.2, 3.0, 3.2])
    A = torch.Tensor([[10, 3, 17, 3.50, 1.7, 8], 
                      [0.05, 10, 17, 0.1, 8, 14],
                      [3, 3.5, 1.7, 10, 17, 8],
                      [17, 8, 0.05, 10, 0.1, 14]])
    
    P = 10**(-4) * torch.Tensor([[1312, 1696, 5569, 124, 8283, 5886],
                                [2329, 4135, 8307, 3736, 1004, 9991],
                                [2348, 1451, 3522, 2883, 3047, 6650],
                                [4047, 8828, 8732, 5743, 1091, 381]])
    
    summand = Variable(torch.zeros(x.shape[0]))
    for i in range(4):
        temp = Variable(torch.zeros(x.shape[0]))
        for j in range(6):
            temp += A[i, j]*(x[:, j] - P[i, j])**2
        
        summand += alpha[i]*torch.exp(-temp)
    
    return summand

def ackley(x):
    #Global max of 0 at origin
    n = x.shape[-1]
    if len(x.shape) == 1:
        x = x.view(1, n)
        
    a = 20; b = 0.2; c=2*math.pi;
    
    term_1 = -a * torch.exp(-b*torch.sqrt(1/n*torch.sum(x**2, dim=1)))
    term_2 = -1*torch.exp(1/n*torch.sum(torch.cos(c*x), dim=1))
    
    return -1*(term_1 + term_2 + a + math.exp(1))