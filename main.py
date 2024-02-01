from admm.DataLoader import *
from admm.model import ADMMNet

def main():
    
    # Dataloading 

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    dataloader = MoonData(n_samples=5000, noise=0.1, device=device)
    X_train, y_train, x_test, y_test = dataloader.get_moon_data_admm()
    X_train =torch.transpose(X_train, 0, 1)
    y_train = torch.transpose(y_train, 0, 1)
    x_test = torch.transpose(x_test, 0, 1)
    y_test = torch.transpose(y_test, 0, 1)

    # Model
    params = {
        "layers" : [(100, 2), (100, 100), (2, 100)],
    }
    net = ADMMNet(params)

    # Training
    net.initialize(X_train, y_train)
    hyper_params = {
        'rho' : 1e-06,
        'tau' : 1e-03,
        'theta' : 1e-03,
        'num_iter': 200
    }
    net.fit(hyper_params, X_train, y_train, x_test, y_test)

    ## Ploting
    net.train_cost[0]
    a1 = net.train_acc
    a2 = net.test_acc
    a3 = net.train_cost

if __name__ == "__main__":
    main()