import torch
import torch.linalg
from torch.optim import Adam
from scipy.sparse.linalg import eigsh
from sklearn.covariance import graphical_lasso

def calcloss(D, Lmd, eps, X, N, V, M):
    """
    Calculate the loss related to the likelihood.

    Parameters
    ----------
    D : Projection matrix (torch.tensor of size V*M)
    Lmd : Precision matrix of latent factors (torch.tensor of size M*M)
    eps : Reciprocal of epsilon
    X : Dataset (torch.tensor of size N*V)
    N : Number of samples
    V : Number of observed variables
    M : Number of latent factors

    Returns
    -------
    float
        The loss related to the likelihood
    """
    loss = torch.mm(torch.t(D), X)
    loss = torch.mm(loss, torch.t(loss))
    loss = torch.sum(loss * Lmd) - torch.trace(loss) * eps
    loss += torch.sum(torch.pow(X,2)) * eps
    loss *= (N**(-1))
    loss -= torch.logdet(Lmd) + (V-M) * torch.log(eps)
    return loss.item()

def calclossrho(Lmd, rho):
    """
    Calculate the loss related to the L1 regularization term.

    Parameters
    ----------
    Lmd : Precision matrix of latent factors (torch.tensor of size M*M)
    rho : Weight of the L1 regularization term

    Returns
    -------
    float
        The loss related to the L1 regularization term
    """
    loss = rho * torch.sum(torch.abs(Lmd))
    return loss.item()

def calcS(D, X, N, V, M):
    """
    Calculate the reciprocal of epsilon

    Parameters
    ----------
    D : Projection matrix (torch.tensor of size V*M)
    X : Dataset (torch.tensor of size N*V)
    N : Number of samples
    V : Number of observed variables
    M : Number of latent factors

    Returns
    -------
    float
        The reciprocal of epsilon
    """
    eps = torch.sum(torch.pow(X,2))
    eps -= torch.sum(torch.pow(torch.mm(torch.t(D),X),2))
    eps = (V-M) * N * torch.reciprocal(eps)
    return eps

def optimizeLmd(D, X, Lmd, N, V, M, rho, mode):
    """
    Calculate the sparse precision matrix of latent factors.

    Parameters
    ----------
    D : Projection matrix (torch.tensor of size V*M)
    X : Dataset (torch.tensor of size N*V)
    Lmd : Precision matrix of latent factors (torch.tensor of size M*M)
        (Return this if the graphical lasso fails.)
    N : Number of samples
    V : Number of observed variables
    M : Number of latent factors
    rho : Weight of the L1 regularization term
    mode : Optimization algorithm for graphical lasso

    Returns
    -------
    Lmd : Precision matrix of latent factors (torch.tensor of size M*M)
    eps : Reciprocal of epsilon
    """

    # Calculate eps, S, and Phi
    eps = calcS(D, X, N, V, M)

    # Graphical LASSO
    try:
        emp_cov = torch.mm(torch.t(D), X)
        emp_cov = torch.mm(emp_cov, torch.t(emp_cov)) * (N**(-1))
        emp_cov = emp_cov.numpy()
        Sgm, Lmd = graphical_lasso(
            emp_cov,
            alpha=rho,
            mode=mode,
        )
        Lmd = torch.from_numpy(Lmd)

    except Exception as e:
        print(e)
        print(f"graphical_lasso Failed.")

    return Lmd, eps


def train(X, M=10, rho=1e-2, learning_rate=1e-3, num_epochs=1000, batch_size=5, mode='cd', dtype='float64', init_D='mean'):
    """
    Train Meta Graphical Lasso model.

    Parameters
    ----------
    X : List of dataset (list of torch.tensor of size N*V)
    M :default: 10 Number of latent factors
    rho :default: 0.01 Weight of the L1 regularization term
    learning_rate :default: 0.001
    num_epochs :default: 1000
    batch_size :default: 5
    mode :default: 'cd' Optimization algorithm for graphical lasso
    dtype :default: 'float64'
    init_D :default: 'mean' Initial value of the projection matrix ('mean': principal components of the concatenated entire dataset, otherwise: random)

    Returns
    -------
    Lmd : List of precision matrix of latent factors (list of torch.tensor of size M*M)
    eps : Reciprocals of epsilon
    D : Projection matrix (torch.tensor of size V*M)
    """

    if dtype == 'float64':
        dtp = torch.float64
    else:
        dtp = torch.float32

    K = len(X) # Number of datasets
    N = [X[k].shape[1] for k in range(K)] # number of samples
    V = X[0].shape[1] # number of variables

    # Initialize precision matrices
    print("initialize Lmd")
    Lmd = []
    for k in range(K):
        Lmd.append(torch.eye(M, dtype=dtp))

    # Calculate initial indicator matrix
    print("initialize D")
    if init_D == 'mean':
        Xall = torch.cat([X[k]/N[k] for k in range(K)], dim=1)
        A,c,B = torch.svd(Xall)
        D = A[:,:M]
    else:
        D = torch.rand(V, M, dtype=dtp)
        P,s,Q = torch.svd(D)
        D = torch.mm(P, torch.t(Q))
    D.requires_grad=False

    # set optimizer for indicator matrix
    optimizer = Adam([D], lr=learning_rate)

    eps = [None for k in range(K)]
    for epoch in range(num_epochs):
        permutation = torch.randperm(K)

        for i in range(0, K, batch_size):
            ind = permutation[i:i+batch_size]

            D_grad = torch.zeros(D.shape, dtype=dtp)

            for ii in range(len(ind)):
                bk = ind[ii]

                # Optimize Lmd
                Lmd[bk], eps[bk], = optimizeLmd(D, X[bk], Lmd[bk], N[bk], V, M, rho, mode)

                # Add to gradient
                omega = Lmd[bk] - torch.eye(M, dtype=dtp) * eps[bk]
                DX = torch.mm(torch.t(D), X[bk])
                DXXD = torch.mm(DX, torch.t(DX))
                grad = torch.mm(DXXD, omega)
                grad = torch.mm(D, grad + torch.t(grad))
                grad -= 2 * torch.mm(X[bk], torch.t(torch.mm(omega, DX)))
                grad *= N[bk] ** (-1)
                D_grad -= grad

            # Update D
            optimizer.zero_grad()
            D.grad = D_grad.detach()
            optimizer.step()

            # Consider Stiefel manifold
            P,s,Q = torch.svd(D)
            D_new = torch.mm(P, torch.t(Q))
            D.data = D_new.detach()

        # Optimize Lmd and calculate loss
        loss_ll = 0
        loss_rho = 0
        for k in range(K):
            loss_ll += calcloss(D, Lmd[k], eps[k], X[k], N[k], V, M)
            loss_rho += calclossrho(Lmd[k], rho)
        loss = loss_ll + loss_rho

        print("epoch: "+str(epoch)+", loss: "+str(loss)+", ll: "+str(loss_ll)+", rho: "+str(loss_rho))

    # Update Lmd for the last D
    for k in range(K):
        Lmd[k], eps[k], = optimizeLmd(D, X[k], Lmd[k], N[k], V, M, rho, mode)

    return Lmd, eps, D

