# main.py

import torch
import pandas as pd
from mglasso.func import train

def main():

    M = 10
    rho = 0.09
#    num_epochs = 5000
    num_epochs = 50
    batch_size = 5

    Xall = pd.read_csv('data/X.csv', header=None)

    X = []
    for k in Xall.iloc[:, 0].unique():
        X.append(torch.tensor(Xall[Xall.iloc[:, 0] == k].iloc[:, 1:].values))

    Lmd, eps, D = train(X, M=M, rho=rho, num_epochs=num_epochs, batch_size=batch_size)

    D = pd.DataFrame(D.numpy())
    D.to_csv('result/D.csv', header=False, index=False)

    for k in range(len(Lmd)):
        Lmd[k] = pd.DataFrame(Lmd[k].numpy())
        Lmd[k].insert(0, 'index', k)
    Lmd = pd.concat(Lmd, ignore_index=True)
    Lmd.to_csv('result/Lmd.csv', header=False, index=False)

    eps = [e.item() for e in eps]
    eps = pd.DataFrame(eps)
    eps.to_csv('result/eps.csv', header=False, index=False)

if __name__ == "__main__":
    main()
