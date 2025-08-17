#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 17 11:23:58 2025

@author: lyh2019
"""
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from tqdm import tqdm

class BMA_port:
    """
    A Bayesian Model Averaging (BMA) class for portfolio optimization.

    Attributes
    ----------
    models : list
        List of BPDS_TVAR model objects
    alpha : float
        Forgetting parameter for updating model weights (default = 1)
    
    Methods
    -------
    fit(T=1, verbose=False)
        Runs BMA for T time steps
    get_returns(data)
        Computes realized returns of BMA and individual models
    """

    def __init__(self, models, alpha=1.0):
        self.models = models
        self.J = len(models)
        self.alpha = alpha
        self.p = 0  # start of forecasting window

    def fit(self, T=1, verbose=False):
        self.m = self.models[0].get_opt_x(0).shape
        self.decisions = np.zeros((T, *self.m))
        self.pi_store = np.zeros((T, self.J))

        pis = np.ones(self.J) / self.J  # initialize equal weights

        iterator = tqdm(range(T)) if verbose else range(T)
        for t in iterator:
            # Update model weights using predictive likelihoods
            pis = self.pi_calc(t, pis)
            self.pi_store[t] = pis

            # Combine model decisions
            opt_x = np.array([self.models[j].get_opt_x(t) for j in range(self.J)])  # (J, q)
            x = pis @ opt_x  # weighted average decision
            self.decisions[t] = x

    def pi_calc(self, t, pis):
        """
        Bayesian model weight update using log predictive likelihoods at time t
        """
        ell = np.array([self.models[j].likelihoods[t, 0] for j in range(self.J)])
        ell_stable = ell - np.max(ell)  # stabilize for exponentiation
        pis_new = pis**self.alpha * np.exp(ell_stable)
        return pis_new / np.sum(pis_new)

    def get_returns(self, data):
        """
        Compute realized returns for individual models and BMA

        data: pd.DataFrame of returns, shape (T, q)
        """
        T = len(self.decisions)
        self.rets = pd.DataFrame(index=data.index[self.p:self.p+T])
        for j in range(self.J):
            self.rets[self.models[j].name] = np.sum(
                self.models[j].opt_x[self.p:self.p+T] * data.iloc[self.p:self.p+T].values,
                axis=1
            )
        self.rets["BMA"] = np.sum(
            self.decisions * data.iloc[self.p:self.p+T].values,
            axis=1
        )
        return self.rets