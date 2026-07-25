import torch
import torch.nn as nn
import numpy as np

class CG_Solver(nn.Module):
    def __init__(self, max_iter=100, tol=1e-6):
        self.max_iter = max_iter
        self.tol = tol

    def solve(self, A_func, b, x0=None):
        # Solve the linear system Ax = b using the Conjugate Gradient method
        # A_func: function that computes the matrix-vector product Ax
        # b: (n_batchs, n_nodes,) right-hand side vector
        # x0: (n_batchs, n_nodes,) initial guess for the solution
        n_batchs, n_nodes = b.size(0), b.size(1)
        if x0 is None:
            x = np.zeros((n_batchs, n_nodes))
        else:
            x = x0.copy()
        
        r = b - A_func(x)  # initial residual, shape (n_batchs, n_nodes)
        p = r.copy()
        rsold = (r * r).sum(1) # shape (n_batchs)

        for i in range(self.max_iter):
            Ap = A_func(p)  # shape (n_batchs, n_nodes)
            alpha = rsold / (p * Ap).sum(1)  # shape (n_batchs)
            x += alpha * p
            r -= alpha * Ap
            rsnew = (r * r).sum(1)
            if np.sqrt(rsnew) < self.tol:
                break
            p = r + (rsnew / rsold) * p
            rsold = rsnew
        
        return x