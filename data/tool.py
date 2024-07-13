import torch
import numpy as np
from sklearn.decomposition import FastICA
from sklearn.covariance import graphical_lasso
from torch.distributions.multivariate_normal import MultivariateNormal

def createLmd(M, sN, alpha):
    """
    Create the precision matrix.

    Parameters
    ----------
    M : Number of latent factors
    sN : Number of samples to create the sparse precision matrix
    alpha : Weight of the L1 regularization term to create the sparse precision matrix

    Returns
    -------
    sSgm : Covariance matrix of latent factors (torch.tensor of size M*M)
    sLmd : Precision matrix of latent factors (torch.tensor of size M*M)
    """

    try:
        H = torch.randn(M, sN, dtype=torch.float64)
        emp_cov = torch.mm(H, torch.t(H)) * (sN**(-1))
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
    """
    Create datasets.

    Parameters
    ----------
    V : Number of observed variables
    M : Number of latent factors
    K : Number of datasets
    N : Number of samples
    sN : Number of samples to create the sparse precision matrix
    alpha : Weight of the L1 regularization term to create the sparse precision matrix
 
    Returns
    -------
    X : List of dataset (list of torch.tensor of size N*V)
    D : Projection matrix (torch.tensor of size V*M)
    Lmd : List of precision matrix of latent factors (list of torch.tensor of size M*M)
    eps : Reciprocals of epsilon
    """

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

