#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 14 17:08:52 2025

@author: lyh2019
"""
import numpy as np
from numpy.linalg import inv, cholesky, LinAlgError
from scipy.stats import multivariate_normal

class TV_VAR:
    def __init__(self, Y, p=1, delta=0.98, lmbda=0.99):
        """
        Time-Varying VAR model with discount factor Kalman filter

        Parameters
        ----------
        Y : ndarray (T x n)
            Time series data (T time points, n variables)
        p : int
            Lag order of the VAR
        delta : float
            State evolution discount factor
        lmbda : float
            Observation variance discount factor
        """
        self.Y = Y
        self.T, self.n = Y.shape
        self.p = p
        self.k = self.n * p + 1  # number of predictors (lags + intercept)
        self.delta = delta
        self.lmbda = lmbda

        # Prepare lagged regressors
        self.X = self._build_lagged_X()

        # Initialize state
        self.theta = np.zeros((self.T, self.k, self.n))  # time-varying coefficients
        self.C = np.zeros((self.T, self.k, self.k))      # state covariance
        self.S = np.zeros((self.T, self.n, self.n))      # observation variance
        self.mu = np.zeros((self.T, self.n))             # one-step ahead mean
        self.loglik = np.zeros(self.T)                   # log-likelihoods

        # BPDS outputs
        self.samples = []
        self.ws = []
        self.likelihoods = []
        self.rs = []
        self.returns = None
        
        # Set priors
        self.theta[0] = np.zeros((self.k, self.n))
        self.C[0] = np.eye(self.k) * 10
        self.S[0] = np.eye(self.n)

    def _build_lagged_X(self):
        T, n, p = self.T, self.n, self.p
        X = []
        for t in range(p, T):
            x_t = [1]  # intercept
            for lag in range(1, p + 1):
                x_t.extend(self.Y[t - lag])
            X.append(x_t)
        return np.array(X)  # shape: (T - p, k)

    def fit(self, n_rep=1000):
        for t in range(self.p, self.T):
            x_t = self.X[t - self.p].reshape(-1, 1)  # shape: (k, 1)
            y_t = self.Y[t].reshape(-1, 1)           # shape: (n, 1)

            theta_t_1 = self.theta[t - 1]
            C_t_1 = self.C[t - 1] / self.delta       # discounted prior
            S_t_1 = self.S[t - 1]

            # Forecast
            R = C_t_1
            Q = x_t.T @ R @ x_t + (1 / self.lmbda)
            Q_inv = inv(Q)

            # Forecast mean
            f = x_t.T @ theta_t_1  # shape: (1, n)
            self.mu[t] = f.flatten()

            # Kalman gain
            A = R @ x_t @ Q_inv    # shape: (k, 1)

            # Update
            e = y_t.T - f          # forecast error
            theta_update = theta_t_1 + A @ e
            C_update = R - A @ x_t.T @ R
            S_update = (1 / self.lmbda) * (S_t_1 + (e.T @ e).reshape(self.n, self.n) / Q)

            self.theta[t] = theta_update
            self.C[t] = C_update
            self.S[t] = S_update

            # Log-likelihood for BPDS
            try:
                logp = multivariate_normal.logpdf(
                    y_t.flatten(), mean=f.flatten(), cov=S_update
                )
            except:
                logp = -1e10  # fallback for numerical errors

            # Store BPDS-compatible outputs
            self.samples.append(self.get_posterior_samples(t, n_rep=n_rep))
            self.ws.append(self.get_opt_x(t))
            self.likelihoods.append(np.array([logp]))
            self.rs.append(np.array([self.mu[t].mean()]))

        # Convert lists to arrays for BPDS use
        self.samples = np.array(self.samples)
        self.ws = np.array(self.ws)
        self.likelihoods = np.array(self.likelihoods)
        self.rs = np.array(self.rs)
        
        # Add realized portfolio returns: w_t applied to Y_{t+1}
        self.returns = np.sum(self.ws[:-1] * self.Y[self.p + 1:], axis=1)
        
    def get_loglikelihoods(self):
        return self.likelihoods

    def get_predictive_mean(self, t):
        return self.mu[t]

    def get_posterior_samples(self, t, n_rep=1000):
        """Draw samples from predictive distribution of y_t given theta_t−1"""
        if t < self.p:
            return np.zeros((n_rep, self.n))
        x_t = self.X[t - self.p].reshape(-1, 1)
        f = x_t.T @ self.theta[t]
        S = self.S[t]
        return np.random.multivariate_normal(mean=f.flatten(), cov=S, size=n_rep)

    def get_opt_x(self, t):
        """
        Placeholder: equal-weight portfolio decision
        """
        return np.ones(self.n) / self.n


# if __name__ == "__main__":   
#     import numpy as np
#     from matplotlib import pyplot as plt
    
#     # Step 1: Simulate data from VAR(1)
#     np.random.seed(42)
#     T = 100
#     n = 2
#     A_true = np.array([[0.8, 0.1],
#                        [0.05, 0.7]])
#     Y = np.zeros((T, n))
#     Y[0] = np.random.normal(size=n)
    
#     for t in range(1, T):
#         Y[t] = A_true @ Y[t-1] + np.random.normal(scale=0.5, size=n)
    
#     # Step 2: Fit TV_VAR model
#     model = TV_VAR(Y, p=1, delta=0.97, lmbda=0.98)
#     model.fit(n_rep=500)
    
#     # Step 3: Inspect outputs
#     print("Shape of predictive samples:", model.samples.shape)       # (T-p, n_rep, n)
#     print("Shape of decision vectors:", model.ws.shape)              # (T-p, n)
#     print("Shape of log-likelihoods:", model.likelihoods.shape)      # (T-p, 1)
#     print("Shape of return targets:", model.rs.shape)                # (T-p, 1)
    
#     # Step 4: Visual check — predictive mean vs. true Y
#     mu = np.array([model.get_predictive_mean(t) for t in range(model.p, T)])
#     true = Y[model.p:]
    
#     fig, axes = plt.subplots(n, 1, figsize=(10, 6), sharex=True)
#     for i in range(n):
#         axes[i].plot(range(model.p, T), true[:, i], label=f"True Y{i+1}")
#         axes[i].plot(range(model.p, T), mu[:, i], label=f"Predicted Y{i+1}", linestyle='--')
#         axes[i].legend()
#         axes[i].set_ylabel(f"Y{i+1}")
#     axes[-1].set_xlabel("Time")
#     plt.suptitle("TV_VAR Predictive Means vs. True Values")
#     plt.tight_layout()
#     plt.show()


