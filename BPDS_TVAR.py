#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  9 23:18:02 2025

@author: lyh2019
"""
import os
os.getcwd()
os.chdir('/Users/lyh2019/Desktop/Bayesian Econometrics/Mike West')


import numpy as np
import pandas as pd
from scipy.optimize import minimize
from tqdm import tqdm  # for progress bar (optional but nice)
from numpy.linalg import inv, cholesky, LinAlgError
from scipy.stats import multivariate_normal
from TV_VAR import TV_VAR


class BPDS_TVAR(object):
    """
    A minimal model class for use in BPDS portfolio examples that is fed in relevant attributes from a full model

    ...

    Attributes
    ----------
    name : str
        name/description of model
    opt_x : array_like
        2D (T x n) array containing n-dimensional model optimal decision made at t-1 targeting time t
    samples : array_like
        3D (T x n_rep x q) containing n_rep samples of the q-dimensional outcome of interest y_{t}, here asset returns between time t-1 and t
    likelihoods : array_like
        1D array of length T containing the log-likelihood of y_{t-1}|D_{t-2}, with the realized value of y_{t-1} observed at t-1
    rs : array_like
        1D array of length T containing the target portfolio return at each time point
    
    Axis 0 is row and Axis 1 is column

    Methods
    -------
    predict(t)
        Accesses relevant samples for time t
    
    get_opt_x(t)
        Accesses model-specific optimal decision for time t
        
    """
    
    def __init__(self, model, name):
        
        self.name = name  # Identifier for the model (e.g., "TVP-VAR(1)", "Baseline")
        self.opt_x = model.ws  # T x n array: optimal decisions at each time step (model-based)
        self.samples = model.samples # T x n_rep x q array: samples of q-dimensional returns at each t
        self.likelihoods = model.likelihoods # T x 1 array: log p(y_t | D_{t−1}) for model weighting
        self.rs = model.rs # T x 1 array: model-specific return targets
        self.returns = model.returns
               
    def predict(self, t):
        return(self.samples[t])

    def get_opt_x(self, t):
        return(self.opt_x[t])
    
        
class BPDS_port(object):

    """
    A class to run and evaluate BPDS for financial portfolio with target return r_t + \tau_1/\tau_2 and constraint \tau_1/\tau_2 < r_t

    ...

    Attributes
    ----------
    models : list
        J-length list of models, including a baseline
    r : float
        Initial portfolio return target
    s_imp : array_like
        Percent improvement targets for setting the BPDS target score, for example s_imp = [1.05, .9] 
        indicates a 5% improvement in return and a 10% reduction in the variance
    alpha : float
        Default is 1, defines how much the BMA weights should decay over time 
    h : int
        Default is None, defines which horizon of interest, if any, to evaluate

    Methods
    -------
    
    fit(self, T = 1, verbose = False)
        Runs BPDS up to time T, set verbose to True to have a progress bar displayed
        
    score_f(self, samples, opt_x, j = None)
        Scores the samples of outcome y_t under decision opt_x for model j
        
    pi_calc(self, models, t, pis)
        Calcualtes the initial probabilities using discounted BMA probabilities
    
    x_calc(self, tau, opt_x, pis)
        Calculates the optimal portoflio using Markowitz optimization using BPDS estimates of the mean and covariance of returns, with target r_t + \tau_1/\tau_2
    
    s_calc(self, scores, pis)
        Calculates the BPDS target score
    
    get_returns(self, data)
        After running fit, run to calculate the returns of the BPDS model (returns DataFrame of model returns + BPDS returns)
    
    _tau_optimize(self,tau, s, pis)
        Internal function to optimize for tau
    
    _update_params(self, pis, t)
        Updates target return
    """
     
    def __init__(self, models, r, s_imp,  alpha=1, h=None):
       
        self.models = models
        self.J = len(models)
        self.r = r
        self.s_imp = s_imp
        self.alpha = alpha
        self.rs = np.zeros(models[0].rs.shape)

    def fit(self, T = 1, verbose = False):
        # T is number of time steps
        self.m = self.models[0].get_opt_x(0).shape  # number of assets
        self.decisions = np.zeros((T, *self.m)) # Store the optimal decisions over time, * unpack the tuple
        self.x = np.zeros(self.m)  # Store a single decision vector q by 1
        b = self.score_f(self.models[0].predict(0), self.models[0].get_opt_x(0)).shape[0] # dimensions of score function
        
        # constraint for tau
        self.tau = np.array([.01, .1]) # initial guess
        self.bounds = tuple([(-25, 25) for i in range(len(self.tau))]) # bounds needed to ensure optimization doesn't wander off
        self.cons =  {'type': 'ineq','fun': lambda x: self.r - x[0]/x[1]}
        self.taus = np.zeros((T,b))
        self.s_store = {"prior": np.zeros((T,b)), "actual": np.zeros((T,b))}

        pis = np.ones(self.J)/(self.J)
        self.pi_store = {"prior": np.zeros((T,self.J)), "BPDS": np.zeros((T,self.J))}
        # model combined score distribution 
        self.score_means = np.zeros((T, b))
        self.score_V = np.zeros((T, b, b))
        
        if verbose:
            l = tqdm(range(T))
        else:
            l = range(T)
            
        for t in l:        
            
            pis = self.pi_calc(self.models, t, pis)
            self.pi_store["prior"][t] = pis
            self._update_params(pis, t)
            
            self.samples = np.array([self.models[j].predict(t) for j in range(self.J)])
            self.opt_x = np.array([self.models[j].get_opt_x(t) for j in range(self.J)])
            self.scores = np.array([self.score_f(self.samples[j], self.opt_x[j]) for j in range(self.J)]) # Compute the score matrix for each model j at time t
            self.s = self.s_calc(self.scores, pis) # mean TARGET scores
            
            s0 = np.multiply(pis[:, None, None], self.scores).sum(axis=0).mean(axis=1) # Mean score vector under the BMA mixture (b,)
            means = np.array([np.vstack(self.scores[i].mean(axis=1)) for i in range(len(pis))]) # Model-Wise Mean Score Vectors (J, b, 1)
            V = np.array([pis[i]*(np.cov(self.scores[i])+means[i]@means[i].T) for i in range(len(pis))]).sum(axis=0) # Covariance matrix of scores under the BMA distribution:

            V = V - np.vstack(s0)@np.vstack(s0).T # np.vstack reshape s0 to a column vector
            self.score_V[t] = V
            self.score_means[t] = s0
            
            if self.cons is not None and self.tau[0]/self.tau[1] > self.r:
                # resets initial value for tau to ensure constraint is satisfied
                # optimization does not start yet
                self.tau[0] = self.tau[0]*self.r/(self.tau[0]/self.tau[1])
                
            if any(self.tau > 5) or any(np.round(self.tau, 5) == 0):
                #if any of the previous values for tau were fairly large/small, try two starting points
                opt = minimize(self._tau_optimize,self.tau,args=(self.s, pis), bounds = self.bounds, constraints = self.cons)
                start2 = np.array([.1, 1])
                if self.cons is not None:
                    start2[0] = start2[0]*self.r/(start2[0]/start2[1])
                opt2 = minimize(self._tau_optimize,start2,args=(self.s, pis), bounds = self.bounds, constraints = self.cons)
                if opt.fun < opt2.fun: # compare which one achieves lower value
                    self.tau = opt.x.reshape(-1) # returns it into 1D arrary
                else:
                    self.tau = opt2.x.reshape(-1)
            else:
                opt = minimize(self._tau_optimize,self.tau,args=(self.s, pis), bounds = self.bounds, constraints = self.cons)
                self.tau = opt.x.reshape(-1)

            a_s = np.mean(np.exp(self.tau@self.scores), axis=1)
            a_deriv = np.mean(self.scores*np.exp(self.tau@self.scores)[:, None, :], axis=2)
            
            self.pi_store["BPDS"][t] = pis*a_s/(pis@a_s) # exponential tilting
            self.x = self.x_calc(self.tau, self.opt_x, pis) # get the optimal weights for portfolios
            
            self.decisions[t] = self.x
            self.taus[t] = self.tau
            self.s_store["prior"][t] = self.s
            self.s_store["actual"][t] = 1/(pis@a_s)*pis@a_deriv
    
    
    def score_f(self, samples, opt_x):
        """
        Computes the score vector for a given model's predictive samples and decision.

        Input:
            samples.shape → (n_rep, q)
                Predictive samples of q-dimensional returns for one model at time t
            opt_x.shape → (q,)
                Decision vector (e.g., portfolio weights) from the same model at time t

        Computation:
            returns.shape → (n_rep,)
                Realized portfolio returns under the model's decision: each sample projected via opt_x
            means.shape → (n_rep,)
                First score component: raw return (xᵗy) for each predictive draw
            sq.shape → (n_rep,)
                Second score component: penalty for deviation from target return (r),
                specifically −½ (xᵗy − r)²

        Output:
            score.shape → (b, n_rep), where b = 2
                Score matrix:
                    - row 0: returns (xᵗy)
                    - row 1: −0.5 × squared error from r

        Purpose:
            Encodes the model's predicted **decision utility** per sample,
            used later for moment-matching (i.e., defining a tilting direction for BPDS)
        """
        returns = np.multiply(samples,opt_x[None, :]).sum(axis=1) # NumPy broadcasting trick, transfer opt_x to (1,q)
        means = returns
        sq = (np.subtract(returns, self.r)**2)
        return(np.array([means, -.5*sq]))
    
    # 4: Bayesian Model Averaging Weights
    def pi_calc(self, models, t, pis):
        """
        Update model weights using exponentially tilted log-predictive likelihoods.

        Formula:
            pi_j(t) = [pi_j(t-1)^alpha * exp(ell_j(t) - min_k ell_k(t))] / Z

        Where:
            - alpha controls decay on prior weight (alpha=1: no decay),
            - ell_j(t) is the log-likelihood: log p(y_t | D_{t-1}) from model j,
            - Z is the normalization constant to ensure weights sum to 1.
            - min_k ell_k(t) ensures numerical stability in exp().

        This implements an adaptive Bayesian model averaging scheme.
        Actually it is the predictive likelihoods
        
        self.alpha takes the AVS-BPS
        """
        for j in range(len(pis)):
            pis[j] = pis[j]**self.alpha*np.exp(self.models[j].likelihoods[t, 0] - min([self.models[i].likelihoods[t, 0] for i in range(len(self.models))] ))
        return (pis/sum(pis))
        
    
    def x_calc(self, tau, opt_x, pis):
        """
        Computes the BPDS-synthesized optimal decision vector x_t at time t.

        Input:
            tau.shape → (2,)
                Tilting vector [τ₀, τ₁] from entropy optimization

            opt_x.shape → (J, q)
                Per-model decision vectors (e.g., portfolio weights)

            pis.shape → (J,)
                Prior model weights from BMA step

            self.scores.shape → (J, b, n_rep)
                Model score arrays — used to weight samples

            self.samples.shape → (J, n_rep, q)
                Predictive samples from each model

        Computation:

            Step 1: Importance weights for each model's samples
                a_s[j] = E_{p_j}[exp(τᵗ s_j(y))] over n_rep samples
                k = 1 / (πᵗ a_s) → normalizing constant

            Step 2: Compute tilted predictive mean vector
                fs[j] = E_{p_j}[y · exp(τᵗ s_j(y))]  (q-dimensional vector)
                f = k × ∑ π_j fs[j]  → overall predictive mean under BPDS

            Step 3: Compute tilted predictive covariance matrix
                Vs[j] = Cov_{p_j}[y · exp(τᵗ s_j(y))]  (q×q matrix)
                V = ∑ k·π_j·(Vs[j] + fs[j]fs[j]ᵗ) − ffᵗ

            Step 4: Solve Markowitz optimization
                - Target return: r + τ₀ / τ₁
                - Objective: minimize xᵗ V x
                - Constraint: xᵗ f = target return
                              sum(x) = 1

        Output:
            opt_x.shape → (q,)
                Final BPDS portfolio weights at time t

        Purpose:
            This function uses the **tilted predictive distribution** to derive
            an **optimal portfolio allocation** that satisfies a return target
            while minimizing variance — a decision-theoretic generalization
            of Bayesian model averaging.
        """
        # compute nomalization constant
        a_s = np.mean(np.exp(tau@self.scores), axis=1)
        k = 1/(pis@a_s)
        opt_x = np.zeros(self.m)
        # compute Tilted Predictive Mean of Returns
        fs = (np.exp(tau@self.scores)[:, :, None]*self.samples).mean(axis=1) # expected return vector under model 
        f = np.vstack(k*(pis@fs)) # shape (q, 1) 

        # Vs = [ (q × q), (q × q), ..., (q × q) ]  ← J elements
        # [:,None] transfer to  (n_rep,q) to facilitate broadcasting
        Vs = [np.cov(
            np.exp(tau@self.scores[j])[ :, None]*self.samples[j], rowvar=False
            ) 
            for j in range(len(pis))
            ]
        
        # compute tilted predictive covariance matrix
        V = np.sum([k*pis[j]* (Vs[j] + np.vstack(fs[j])@np.vstack(fs[j]).T) for j in range(len(pis))], axis=0)
        V = V-f@f.T
        # solve optimization
        opt_x = self.target_return_port(self.r+tau[0]/tau[1], f, V).T[0]
            
        return(opt_x)
        
    def s_calc(self, scores, pis):
        """
        Computes the BPDS target score vector (s) via model-weighted average of sample scores.

        Input:
            scores.shape → (J, b, n_rep)
                Score matrices from all J models:
                    - J: number of models
                    - b: number of score components (e.g., return, penalty)
                    - n_rep: number of predictive samples per model

            pis.shape → (J,)
                Current model weights (prior or BMA-style), summing to 1

        Computation:
            - scores * pis[:, None, None] → broadcast model weights across all scores
            - sum(axis=0) → aggregate across models
            - mean(axis=1) → average over predictive samples (Monte Carlo expectation)
            - Multiply by self.s_imp → apply desired improvements (e.g., ↑return, ↓risk)

        Output:
            s.shape → (b,)
                Target expected score vector under the tilted predictive distribution.
                This is the moment condition: we seek τ such that E_τ[s(y)] ≈ s.

        Purpose:
            This is the **target moment** the BPDS optimization (entropic tilting) will try to match.
            It defines what kind of improvement you want over standard model averaging (BMA).
        """
        s = np.multiply(pis[:, None, None], scores).sum(axis=0).mean(axis=1)*self.s_imp
        return(s)
        
    def get_returns(self, data):
        self.rets = pd.DataFrame(index = data.index[1:len(self.decisions)])
        for j in range(len(self.models)):
            self.rets[self.models[j].name] = self.models[j].returns
        self.rets["BPDS"]= (self.decisions[:-1]*data.iloc[1:len(self.decisions)]).sum(axis=1)
        return(self.rets)
        
    def _tau_optimize(self,tau, s, pis):
        """
        Objective function for optimizing the tilting vector τ = [τ₀, τ₁].

        This function computes the discrepancy between:
            - The target score vector s (set by BPDS via s_calc)
            - The expected score vector under the tilted predictive mixture

        The optimization goal is:
            min_τ || g(τ) ||₁
        where:
            g(τ) = s - E_τ[s(y)]
            → This is a moment-matching objective: we want the expected score under the tilted
              predictive distribution to match the target score vector `s`.

        Input:
            tau.shape → (2,)
                Current candidate tilting vector [τ₀, τ₁]
            s.shape → (b,)
                Target expected score vector, often s = s_BMA * s_imp
            pis.shape → (J,)
                Current model weights (from pi_calc)

        Computation:
            - a_s.shape → (J,)
                Normalization terms (importance weights) for each model under τ:
                a_s[j] = E_{p_j}[exp(τᵗ s_j(y))]

            - a_deriv.shape → (J, b)
                Weighted score expectations under τ for each model:
                a_deriv[j] = E_{p_j}[s_j(y) · exp(τᵗ s_j(y))]

            - g.shape → (b,)
                Moment-matching gap:
                g = s - E_τ[s(y)] = s - (∑ π_j a_deriv[j]) / (∑ π_j a_s[j])

        Output:
            Objective value → scalar
                L1 norm: sum(abs(g)), measuring how far off the current τ is from achieving the desired score vector s

        Purpose:
            This is the core optimization objective used in BPDS:
            Find τ such that the exponentially tilted predictive mixture yields an expected score vector ≈ s.
        """
        a_s = np.mean(np.exp(tau@self.scores), axis=1) # model-weighted importance weights
        a_deriv = np.mean(self.scores*np.exp(tau@self.scores)[:, None, :], axis=2)
        g = s - 1/(pis@a_s)*pis@a_deriv
        return(sum(abs(g))) 
    
    def _update_params(self, pis, t):
        """
        Input:
            pis.shape → (J,)
                Model weights at time t (updated via pi_calc)
            self.models[j].rs[t] → scalar
                Model j's internal return target at time t
        
        Computation:
            - Computes the weighted average of model-specific return targets at time t
            - Stores the result in self.r for use in scoring and optimization
            - Also records the return trajectory in self.rs[t] for reference
        
        Output:
            self.r → scalar
                BPDS-combined portfolio return target at time t
            self.rs[t] → scalar
                Time series record of BPDS target returns
        """
        self.r = pis@np.array([self.models[j].rs[t] for j in range(len(self.models))])
        self.rs[t] = self.r

    def target_return_port(self, r_target, f, V):
        """
        Solves the Markowitz portfolio optimization using scipy.optimize.minimize.
        
        Parameters
        ----------
        r_target : float
        Target return.
        f : np.array, shape (q, 1)
        Predictive mean vector of returns.
        V : np.array, shape (q, q)
        Covariance matrix of returns.
        
        Returns
        -------
        x_opt : np.array, shape (q, 1)
        Optimal portfolio weights.
        """
        q = f.shape[0]
        f = f.flatten()  # Ensure shape (q,)
        
        # Initial guess: equally weighted
        x0 = np.ones(q) / q
        
        # Objective: portfolio variance
        def objective(x):
            return x @ V @ x
        
        # Constraints: return = target, weights sum to 1
        constraints = [
        {'type': 'eq', 'fun': lambda x: np.sum(x) - 1},
        {'type': 'eq', 'fun': lambda x: x @ f - r_target}
        ]
        
        # No bounds unless you want to restrict (e.g., long-only)
        result = minimize(objective, x0, constraints=constraints)
        
        if not result.success:
            raise ValueError(f"Optimization failed: {result.message}")
        
        return result.x.reshape(-1, 1)    





# if __name__ == "__main__":   
# Simulate or load your data
np.random.seed(123)
T, n = 100, 2
A_true = np.array([[0.8, 0.1], [0.05, 0.7]])
Y = np.zeros((T, n))
Y[0] = np.random.normal(size=n)

for t in range(1, T):
    Y[t] = A_true @ Y[t - 1] + np.random.normal(scale=0.5, size=n)

# Create multiple TV_VAR models with different discount factors
delta_vals = [0.95, 0.97, 0.99]
models = []

for delta in delta_vals:
    m = TV_VAR(Y, p=1, delta=delta, lmbda=0.98)
    m.fit(n_rep=500)
    models.append(m)


mods = [BPDS_TVAR(model=m, name=f"TV_VAR_δ{m.delta}") for m in models]

BPDS = BPDS_port(mods, r=0.1, s_imp=np.array([1.05, -0.9]), alpha=0.9)
BPDS.fit(T=len(Y) - m.p, verbose=True)

import pandas as pd

# Assume Y.shape = (T, n), e.g., (100, 2)
data_pct = pd.DataFrame(Y, columns=[f"Asset{i+1}" for i in range(Y.shape[1])])

returns_bpds = BPDS.get_returns(data_pct)


import matplotlib.pyplot as plt

returns_bpds.plot(figsize=(12, 6))
plt.title("Realized Returns Over Time")
plt.xlabel("Time")
plt.ylabel("Portfolio Return")
plt.grid(True)
plt.legend(title="Model")
plt.tight_layout()
plt.show()










"""
Example Code after reading in relevant data and models

Attributes
----------
mods : list
    list of BPDS_TVAR models, including a baseline at index 0
r : float
    Initial portfolio return target
s_imp : array_like
    Percent improvement targets for setting the BPDS target score, for example s_imp = [1.05, .9] 
    indicates a 5% improvement in return and a 10% reduction in the variance
alpha : float
    Default is 1, defines how much the BMA weights should decay over time 
data_pct: pandas DataFrame
    Contains asset returns up to time T
"""

BPDS = BPDS_port(mods, r=.1, s_imp=np.array([1.05, .9]), alpha=.8)
BPDS.fit(T = len(data_pct), verbose = True)
returns = BPDS.get_returns(data_pct)
        
