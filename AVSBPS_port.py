#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 17 11:24:33 2025

@author: lyh2019
"""
import numpy as np
import pandas as pd
from tqdm import tqdm

class AVSBPS_port:
    """
    Adaptive Variational Synthesis BPS for portfolio decision-making.

    This version weights models based on their average decision score,
    using exponentially weighted updates (no τ optimization).

    Attributes
    ----------
    models : list
        List of BPDS_TVAR-wrapped models
    s_weight : array_like
        Vector of weights to apply to score components (e.g., [1.05, -0.9])
    alpha : float
        Exponential decay parameter for prior weights (default = 1)
    p : int
        Forecasting start index (default = 0)

    Methods
    -------
    fit(T, verbose)
        Runs AVSBPS procedure over T time steps
    get_returns(data)
        Applies final decisions to return data and computes realized returns
    """

    def __init__(self, models, s_weight=np.array([1.05, -0.9]), alpha=1.0):
        self.models = models
        self.J = len(models)
        self.s_weight = s_weight
        self.alpha = alpha
        self.p = 0  # forecasting window start index

    def fit(self, T=1, verbose=False):
        self.m = self.models[0].get_opt_x(0).shape
        self.decisions = np.zeros((T, *self.m))
        self.pi_store = np.zeros((T, self.J))
        self.score_store = np.zeros((T, self.J))  # track scalar scores for diagnostics

        pis = np.ones(self.J) / self.J  # initialize equal weights
        iterator = tqdm(range(T)) if verbose else range(T)

        for t in iterator:
            # Compute scalar score = weighted sum of score components (per model)
            scores = np.array([
                self.s_weight @ self.models[j].score_f(
                    self.models[j].predict(t),
                    self.models[j].get_opt_x(t)
                ).mean(axis=1)
                for j in range(self.J)
            ])
            self.score_store[t] = scores

            # Update model weights
            scores = scores - np.max(scores)  # numerical stabilization
            pis = pis**self.alpha * np.exp(scores)
            pis /= np.sum(pis)
            self.pi_store[t] = pis

            # Combine decisions
            opt_x = np.array([self.models[j].get_opt_x(t) for j in range(self.J)])
            self.decisions[t] = pis @ opt_x

    def get_returns(self, data):
        """
        Compute realized returns for individual models and AVSBPS ensemble.

        Parameters
        ----------
        data : pd.DataFrame
            Time-indexed matrix of returns (T x q)

        Returns
        -------
        self.rets : pd.DataFrame
            Realized returns for each model and AVSBPS ensemble
        """
        T = len(self.decisions)
        self.rets = pd.DataFrame(index=data.index[self.p:self.p+T])

        for j in range(self.J):
            self.rets[self.models[j].name] = np.sum(
                self.models[j].opt_x[self.p:self.p+T] * data.iloc[self.p:self.p+T].values,
                axis=1
            )

        self.rets["AVSBPS"] = np.sum(
            self.decisions * data.iloc[self.p:self.p+T].values,
            axis=1
        )
        return self.rets

AVSBPS = AVSBPS_port(mods, s_weight=np.array([1.05, -0.9]), alpha=0.8)
AVSBPS.fit(T=len(data_pct), verbose=True)
returns_avs = AVSBPS.get_returns(data_pct)