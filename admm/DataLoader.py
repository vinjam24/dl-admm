from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
import torch
import numpy as np

class MoonData():
    
    def __init__(self, n_samples=1000, noise=0.1, device='cpu', random_state=42):
        self.n_samples = n_samples
        self.noise = noise
        self.device = device
        self.random_state = random_state
    
    def get_moon_data_admm(self):

        X, y = make_moons(n_samples=self.n_samples, noise=self.noise, random_state=self.random_state)
        len_classes = 2
        y_h = np.zeros((len(y), 2))
        for i in range(2):
            class_vec = np.zeros(len_classes)
            class_vec[i] = 1
            y_h[np.where(y==i)] = class_vec

        # Train test split
        X_train, x_test, y_train, y_test = train_test_split(X, y_h, test_size=0.3, random_state=42)
        X_train = torch.as_tensor(X_train, dtype=torch.float32, device=self.device)
        y_train = torch.as_tensor(y_train, dtype=torch.float32, device=self.device)
        x_test = torch.as_tensor(x_test, dtype=torch.float32, device=self.device)
        y_test = torch.as_tensor(y_test, dtype=torch.float32, device=self.device)

        return X_train, y_train, x_test, y_test