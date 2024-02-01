import numpy as np
import torch
import time
from utils.helpers import *
from admm.metrics import *

class ADMMNet():

    def __init__(self, params, device='cpu'):
       
        seed_num = 18
        torch.random.manual_seed(seed=seed_num)
        
        hidden_layers = params['layers']
        self.W = []
        self.b = []
        self.z = []
        self.a = []
        self.u = None
        self.n_layers = len(hidden_layers)
        
        self.train_acc = [0]
        self.test_acc = [0]
        self.train_cost = []
        self.test_cost = []
        self.objective_value = []
        
        for layer in hidden_layers:
            torch.random.manual_seed(seed=seed_num)
            self.W.append(torch.normal(size=layer, mean=0, std=0.1, device=device))
            torch.random.manual_seed(seed=seed_num)
            self.b.append(torch.normal(size=(layer[0], 1), mean=0, std=0.1, device=device))
        
    def initialize(self, x_train, label):
        for i in range(self.n_layers):
            if(i == self.n_layers-1):
                imask = torch.eq(label, 0)
                self.z.append(torch.where(imask, -torch.ones_like(label), torch.ones_like(label)))
            elif(i==0):             
                self.z.append(torch.matmul(self.W[i], x_train) + self.b[i])
                self.a.append(relu(self.z[-1]))
            else:
                self.z.append(torch.matmul(self.W[i], self.a[i-1]) + self.b[i])
                self.a.append(relu(self.z[-1]))

        self.u = torch.zeros(self.z[-1].shape, device=device)    
    
    
    def fit(self, hyper_params, x_train, y_train, x_test, y_test):
        
       
        
        ## Params
        rho = hyper_params["rho"]
        theta = hyper_params["theta"]
        tau = hyper_params["tau"]        
        num_iter = hyper_params["num_iter"]
        
        linear_r = np.ones(num_iter)
        self.objective_value = np.ones(num_iter)
        self.train_acc = np.zeros(num_iter+1)
        self.train_cost = np.ones(num_iter+1)
        self.test_acc = np.zeros(num_iter+1)
        self.test_cost = np.ones(num_iter)
        ## Backward
        (_, self.train_cost[0]) = get_accuracy(self.W, self.b, x_train, y_train)
        
        for j in range(num_iter):
            pre = time.time()
            for i in range(self.n_layers-1, -1, -1):
                if(i!=0):
                    if(i==self.n_layers-1):
                        self.z[i] = update_zl(self.a[i-1], self.W[i], self.b[i], y_train, self.z[i], self.u, rho)
                        self.b[i] = update_b(self.a[i-1], self.W[i], self.z[i], self.b[i], self.u, rho)
                        self.W[i] = update_W(self.a[i-1], self.b[i], self.z[i], self.W[i], self.u, rho, theta)
                        self.a[i-1] = update_a(self.W[i], self.b[i], self.z[i], self.z[i-1], self.a[i-1], self.u, 0, rho, tau)
                    else:
                        self.z[i] = update_z(self.a[i-1], self.W[i], self.b[i], self.a[i], 0, 0, rho)
                        self.b[i] = update_b(self.a[i-1], self.W[i], self.z[i], self.b[i], 0, rho)
                        self.W[i] = update_W(self.a[i-1], self.b[i], self.z[i], self.W[i], 0, rho, theta)
                        self.a[i-1] = update_a(self.W[i], self.b[i], self.z[i], self.z[i-1], self.a[i-1], 0, 0, rho, tau)

                else:
                    self.z[i] = update_z(x_train, self.W[i], self.b[i], self.a[i], 0, 0, rho)
                    self.b[i] = update_b(x_train, self.W[i], self.z[i], self.b[i], 0, rho)
                    self.W[i] = update_W(x_train, self.b[i], self.z[i], self.W[i], 0, rho, theta)
            
            ## Forward
            for i in range(self.n_layers):
                if(i==0):
                    self.W[i] = update_W(x_train, self.b[i], self.z[i], self.W[i], 0, rho, theta)
                    self.b[i] = update_b(x_train, self.W[i], self.z[i], self.b[i], 0, rho)
                    self.z[i] = update_z(x_train, self.W[i], self.b[i], self.a[i], 0, 0, rho)
                    self.a[i] = update_a(self.W[i+1], self.b[i+1], self.z[i+1], self.z[i], self.a[i], 0, 0, rho, tau)
                elif(i==self.n_layers-1):
                    self.W[i] = update_W(self.a[i-1], self.b[i], self.z[i], self.W[i], self.u, rho, theta)
                    self.b[i] = update_b(self.a[i-1], self.W[i], self.z[i], self.b[i], self.u, rho)
                    self.z[i] = update_zl(self.a[i-1], self.W[i], self.b[i], y_train, self.z[i], self.u, rho)
                else:
                    self.W[i] = update_W(self.a[i-1], self.b[i], self.z[i], self.W[i], 0, rho, theta)
                    self.b[i] = update_b(self.a[i-1], self.W[i], self.z[i], self.b[i], 0, rho)
                    self.z[i] = update_z(self.a[i-1], self.W[i], self.b[i], self.a[i], 0, 0, rho)
                    self.a[i] = update_a(self.W[i+1], self.b[i+1], self.z[i+1], self.z[i], self.a[i], self.u, 0, rho, tau)
                    

            self.u = self.u + rho * (self.z[-1] - torch.matmul(self.W[-1], self.a[-1]) - self.b[-1])
            r = []
            for i in range(self.n_layers):
                if(i == 0):
                    r.append(torch.sum((self.z[i] - torch.matmul(self.W[i], x_train) - self.b[i]) * (self.z[i] - torch.matmul(self.W[i], x_train) - self.b[i])))
                else:
                    r.append(torch.sum((self.z[i] - torch.matmul( self.W[i], self.a[i-1]) - self.b[i]) * (self.z[i] - torch.matmul(self.W[i], self.a[i-1]) - self.b[i])))

            linear_r[j] = r[-1]

            obj, loss = objective(x_train, y_train, self.W, self.b, self.z, self.a, self.u, rho)
            self.objective_value[j] = obj.cpu().numpy()
            (self.train_acc[j+1], self.train_cost[j+1]) = get_accuracy(self.W, self.b, x_train, y_train)
            (self.test_acc[j+1], self.test_cost[j]) = get_accuracy(self.W, self.b, x_test, y_test)
            
            print("Epoch: {} | Lagrangian Objective: {:.2f} | Train Loss: {:.2f} | Test loss: {:.2f} | Train accuracy: {:.2f} | Test Accuracy: {:.2f}".format(
                j+1, obj, self.train_cost[j], self.test_cost[j], self.train_acc[j], self.test_acc[j]
            ))
            print("=====================================================================================================================")
                
        
            
            
        