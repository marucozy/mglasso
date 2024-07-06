# main.py

import torch
import pandas as pd
from data.tool import create

def main():

    V = 100
    M = 10
    K = 10
    N = 200
    sN = 20
    alpha = 0.3

    X,D,Lmd,eps = create(V, M, K, N, sN, alpha)

    for k in range(len(X)):
        X[k] = pd.DataFrame(X[k].numpy())
        X[k].insert(0, 'index', k)
    X = pd.concat(X, ignore_index=True)
    X.to_csv('data/X.csv', header=False, index=False)

    D = pd.DataFrame(D.numpy())
    D.to_csv('data/D.csv', header=False, index=False)

    for k in range(len(Lmd)):
        Lmd[k] = pd.DataFrame(Lmd[k].numpy())
        Lmd[k].insert(0, 'index', k)
    Lmd = pd.concat(Lmd, ignore_index=True)
    Lmd.to_csv('data/Lmd.csv', header=False, index=False)

    eps = torch.cat(eps, dim=0)
    eps = pd.DataFrame(eps.numpy())
    eps.to_csv('data/eps.csv', header=False, index=False)

if __name__ == "__main__":
    main()
