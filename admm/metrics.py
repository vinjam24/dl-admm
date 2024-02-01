import torch
from utils.helpers import *

def get_accuracy(W, b, X, y):
    nums = y.shape[1]
    z = []
    a = []
    for i in range(len(W)):
        if(i == 0):
            z.append(torch.matmul(W[i], X) + b[i])
        else:
            a.append(relu(z[-1]))
            z.append(torch.matmul(W[i], a[i-1]) + b[i])
    
    cost = cross_entropy_with_softmax(y, z[-1]) / nums
    actual = torch.argmax(y, dim=0)
    pred = torch.argmax(z[-1], dim=0)
    return (torch.sum(torch.eq(pred, actual)) / nums, cost)    


# return the value of the augmented Lagrangian
def objective(x_train, y_train, W, b, z, a, u, rho):
    r = []
    for i in range(len(W)):
        if(i==0):
            r.append(torch.sum((z[i] - torch.matmul(W[i], x_train) - b[i]) * (z[i] - torch.matmul(W[i], x_train) - b[i])))
        else:
            r.append(torch.sum((z[i] - torch.matmul(W[i], a[i-1]) - b[i]) * (z[i] - torch.matmul(W[i], a[i-1]) - b[i])))
    
    loss = cross_entropy_with_softmax(y_train, z[-1])
    
    obj = loss + torch.trace(torch.matmul(z[-1] - torch.matmul(W[-1], a[-1]) - b[-1], torch.transpose(u,0,1)))
    
    for i in range(len(r)):
        obj = obj + (rho/2)*r[i]
    cum = 0
    for i in range(len(a)):
        obj = obj + (rho/2)*torch.sum((a[i] - relu(z[i])) * (a[i] - relu(z[i])))
    return obj, loss