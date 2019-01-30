"""
    Data generation - Monte Carlo simulation Simulations
        - Additive model
        - Multiplicative model


    Assumed Causal diagram:
        U       -       An unmeasured confounder
        Z       -       Instrumental variable
        (X,Y)   -       The exposure-outcome pair

    Graph structure:
        U -> X (alpha1)
        U -> Y (beta1)
        X -> Y (beta2)
        Z -> U (gamma1)
        Z -> X (alpha2)

    The true "exposure effect" and target of estimation is beta2.

    For simplicity and to reflect a common study framework, all variables are binary.

    The measured covariate Z may act as a confounder or as an instrumental variable for the exposure-outcome pair (X, Y).
    Note that Z is not a perfect instrument because it is associated with the unmeasured confounder U through gamma1
    However, by varying the value of gamma1, we can explore the impacts of conditioning on Z when it is a perfect instrument and when it is a near-instrument or confounder.
    As shown by Pearl (24), bias amplification may result even when the conditioning variable is not a perfect instrument.
    Consider relatively large values of gamma1 to compare the risks of adjusting for an IV with the benefits of adjusting for a real confounder.

"""

##################################
## Packages ##
import logging
from collections import defaultdict, Counter
from datetime import datetime

import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.calibration import calibration_curve
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict, cross_val_score

print(__doc__)


class MonteCarloSimulation(object):

    def __int__(self, n_simulations=2500, dataset_size=10000):

        print("Start Monte Carlo Simulation ..", datetime.now())

        self.n_simulations = n_simulations
        self.dataset_size = dataset_size

        #TODO: handle these params, we should play around with the values of the m
        self.gamma0 = .3
        self.gamma1 = .0 # 0.006, 0.06, 0.24, 0.6
        self.alpha0 = .3
        self.alpha1 = .0 # 0.06, 0.18, 0.33
        self.alpha2 = 0 # 0.06, 0.18, 0.33
        self.beta0 = .2 #.01
        self.beta1 = .0 # 0.08, 0.36, 0.5
        self.beta2 = .0 # .2

        self.additive_data = self.sim_additive_model()

        self.multi_data = self.sim_multiplic_model()

        self.save_data_to_file() # TODO: to implment

        print("Finish Monte Carlo Simulation ..", datetime.now())

    def sim_additive_model(self):

        """
        Simulated data under an additive model (additive risk)

        The goal of estimation: the risk difference in outcome between levels of exposure.

        Simulate n_datasets where each dataset consists of 4 binary variables:
            Z, with P(Z=1) = .5
            U, X and Y such that
                P(U=1|Z) = gamma0 + gamma1 * Z
                P(X=1|U,Z) = alpha0 + alpha1 * U + alpha2 * Z
                P(Y=1|X,U) = beta0 + beta1 * X + beta2 * U

        Notes:
            1. Order of variables simulated is important so that the risk of outcome would depend directly on U and X and indirectly on Z.
            2. gamma0, alpha0 and beta0 define the baseline prevalence of each variable and each effect parameter may be interpreted as a risk difference

        :return: data: numpy ndarray, matrix of matrices
        """
        print("Start Additive Simulation ..", datetime.now())

        # Empty matrix of matrices, will hold all the simulated datasets (each dataset is a matrix with 3 columns)
        data = np.empty((self.n_simulations, self.dataset_size, 3), int)

        # Number of trials, number of observations
        num_trials = 1
        n_obs = self.dataset_size

        def generate_data():
            # Draw samples from a binomial distribution
            z = np.random.binomial(n=num_trials, p=.5, size=n_obs)
            u = np.random.binomial(n=num_trials, p=self.gamma0 + self.gamma1 * z, size=n_obs)
            x = np.random.binomial(n=num_trials, p=self.alpha0 + self.alpha1 * u + self.alpha2 * z, size=n_obs)
            y = np.random.binomial(n=num_trials, p=self.beta0 + self.beta1 * u + self.beta2 * x, size=n_obs)
            # Merge z,x, and y into one matrix
            sample = np.stack((z,x,y), axis=1)
            return sample

        for sim in range(self.n_simulations):
            # Store every dataset as an element in the matrix
            data[sim,] = generate_data()

        print("Finish Additive Simulation ..", datetime.now())

        return data

    def sim_multiplic_model(self):
        """
        Simulated data under a multiplicative model (multiplicative risk)

        The goal of estimation is considered to be the risk ratio for the outcome according to level of exposure.

        Simulate n_datasets where each dataset consists of 4 binary variables:
            Z, with P(Z=1) = .5
            U, X and Y such that
                P(U=1|Z) = gamma0 * (gamma1 ^ Z)
                P(X=1|U,Z) = alpha0 * (alpha1 ^ U) * (alpha2 ^ Z)
                P(Y=1|X,U) = beta0 * (beta1 ^ X) * (beta2 ^ U)

        :return: data: numpy ndarray, matrix of matrices
        """
        print("Start Multiplicative Simulation ..", datetime.now())

        # Empty matrix of matrices, will hold all the simulated datasets (each dataset is a matrix with 3 columns)
        data = np.empty((self.n_simulations, self.dataset_size, 3), int)

        # Number of trials, number of observations
        num_trials = 1
        n_obs = self.dataset_size

        def generate_data():
            # Draw samples from a binomial distribution
            z = np.random.binomial(n=num_trials, p=.5, size=n_obs)
            u = np.random.binomial(n=num_trials, p=self.gamma0 * np.power(self.gamma1, z), size=n_obs)
            x = np.random.binomial(n=num_trials, p=self.alpha0 * np.power(self.alpha1, u) * np.power(self.alpha2, z), size=n_obs)
            y = np.random.binomial(n=num_trials, p=self.beta0 * np.power(self.beta1, u) * np.power(self.beta2, x), size=n_obs)
            # Merge z,x, and y into one matrix
            sample = np.stack((z,x,y), axis=1)
            return sample

        for sim in range(self.n_simulations):
            # Store every dataset as an element in the matrix
            data[sim,] = generate_data()

        print("Finish Multiplicative Simulation ..", datetime.now())
        return data

    def save_data_to_file(self):
        print("Save datasets to file ..." , datetime.now())
        np.save(file='PATH_1', arr=self.additive_data)
        np.save(file='PATH_2', arr=self.multi_data)

        return None


# ----------------------------------------------------------------------------
if __name__ == "__main__":
    MonteCarloSimulation().__int__()

