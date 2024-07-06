import torch
import numpy as np
from sklearn.decomposition import FastICA
from sklearn.covariance import graphical_lasso
from torch.distributions.multivariate_normal import MultivariateNormal

def createLmd(M, N, alpha):

    try:
        H = torch.randn(M, N, dtype=torch.float64)
        emp_cov = torch.mm(H, torch.t(H)) * (N**(-1))
        emp_cov = emp_cov.numpy()
        sSgm, sLmd = graphical_lasso(
            emp_cov,
            alpha=alpha,
            mode='cd',
        )
        sLmd = torch.from_numpy(sLmd)
        sSgm = torch.from_numpy(sSgm)

    except Exception as e:
        print(e)
        print(f"graphical_lasso Failed.")

    return sSgm, sLmd


def create(V, M, K, N, sN, alpha):

    D = torch.rand(V, M, dtype=torch.float64)
    P,s,Q = torch.svd(D)
    D = torch.mm(P, torch.t(Q))

    X = []
    Lmd = []
    eps = []
    for k in range(K):
                        
        sSgm, sLmd = createLmd(M, sN, alpha)
        Lmd.append(sLmd)
        eigenvalues, _ = torch.linalg.eigh(Lmd[k])
        if (eigenvalues > 0).all():
            pass
        else:
            print("eigenvalues are not all non-zero")

        eps.append(torch.abs(torch.randn(1, dtype=torch.float64)) * 10.)

        # Create a MultivariateNormal distribution
        mean = torch.zeros(V, dtype=torch.float64)  # Mean of the distribution
#        precision_matrix = torch.tensor(torch.mm(D, torch.mm(Lmd[k], D.t())))
        precision_matrix = torch.mm(D, torch.mm(Lmd[k], D.t()))
        precision_matrix += eps[k] * (torch.eye(V, dtype=torch.float64) - torch.mm(D, D.t()))
        multivariate_normal = MultivariateNormal(mean, precision_matrix=precision_matrix)

        # Generate samples from the multivariate Gaussian distribution
        cX = multivariate_normal.sample((N,))
        cX = cX.t()

        X.append(cX)

    return(X, D, Lmd, eps)

