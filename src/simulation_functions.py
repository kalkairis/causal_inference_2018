import os
import sys
from os import path

import numpy as np

try:
    from src.simulation_iterator import SimulationIterator
except:
    from simulation_iterator import SimulationIterator


def kang_shafer_data(size=(1,), random_seed=0, X_observed=True):
    """
        Generate Y_true from Kang-Schafer Data Generating Process
        
        Parameters
        ----------
        size : size of output arrays
        random_seed : numpy random seed. default=0
        X_observed : Specify whether Xs are "not observed" , but a nonlinear transformation of them is
        
        Inside
        ------
        Xs are auxiliary random variables, true_propensity, A is treatment variable
        
        Returns
        -------
        A: binary treatment assignment
        Y_true: Simulated Y
    
        References
        ----------
        Joseph DY Kang, Joseph L Schafer, et al. Demystifying double robustness: A comparison of alternative strategies
        for estimating a population mean from incomplete data. Statistical science, 22(4):523–539, 2007.
        
    """
    np.random.seed = random_seed
    X1 = np.random.normal(0, 1, size=size)
    X2 = np.random.normal(0, 1, size=size)
    X3 = np.random.normal(0, 1, size=size)
    X4 = np.random.normal(0, 1, size=size)

    true_propensity = 1 / (1 + np.exp(X1 - 0.5 * X2 + 0.25 * X3 + 0.1 * X4))
    t = (true_propensity >= 0.5).astype(int)
    # print(X1, X2, X3, X4, true_propensity, A)

    E_Y = 210 + t + 27.4 * X1 + 13.78 * X2 + 13.7 * X3 + 13.7 * X4

    y = np.random.normal(E_Y, 1)

    if X_observed:
        return X1, X2, X3, X4, t, y
    else:  # Xs are "not observed" , but a nonlinear transformation of them is
        X1_t = np.exp(X1 / 2)
        X2_t = (X2 / (1 + np.exp(X1))) + 10
        X3_t = (((X1 * X3) / 25) + 0.6) ** 3
        X4_t = (X2 + X4 + 20) ** 2
        return X1_t, X2_t, X3_t, X4_t, t, y


def get_ACIC_2019_datasets(high_dim=True, test=True):
    """
    This function allows to run over the ACIC 2019 datasets
    :param high_dim: Return high dimentional data (if False will return low dimensional data)
    :param test: Return test data or full data
    :return: An iterator that will return the dataframe at each iteration
    """
    urls = {
        'test': {'high': ['TestDatasets_highD',
                          'https://www.google.com/url?q=https%3A%2F%2Fwww.dropbox.com%2Fs%2F7st5ttdihk6dzfz%2FTestDatas\
                          ets_highD_Dec28.zip%3Fdl%3D1&sa=D&sntz=1&usg=AFQjCNFQrGvw1sdDNVQK5mNCgX_DxPXUSw'],
                 'low': ['TestDatasets_lowD',
                         'https://www.google.com/url?q=https%3A%2F%2Fwww.dropbox.com%2Fs%2Fqaj6fjbzorzmwpp%2FTest\
                         Datasets_lowD_Dec28.zip%3Fdl%3D1&sa=D&sntz=1&usg=AFQjCNE91-3NWEzwQ2yeziB0pJdRxO865g']},
        'full': {'high': ['high_dimensional_datasets',
                          'https://www.google.com/url?q=https%3A%2F%2Fwww.dropbox.com%2Fs%2Fk2k1cs42i3pzkuu%2Fhigh\
                          _dimensional_datasets.zip%3Fdl%3D0&sa=D&sntz=1&usg=AFQjCNHYWOhPeNuu5_6AgwyxQ1A75PIOMQ'],
                 'low': ['low_dimensional_datasets',
                         'https://www.google.com/url?q=https%3A%2F%2Fwww.dropbox.com%2Fs%2Fg0elnbfmhbf7rr3%2Flow\
                         _dimensional_datasets.zip%3Fdl%3D0&sa=D&sntz=1&usg=AFQjCNHuy9b9ViStuxXpP3THSJtY2SX5gg']}
    }

    dir_name, url = urls['test' if test else 'full']['high' if high_dim else 'low']
    out_dir = path.realpath(
        path.join(path.join(path.join(path.join(path.abspath(path.dirname(sys.argv[1])), '..'), 'Data'), 'ACIC'),
                  dir_name))
    if not os.path.exists(out_dir):
        raise Exception(f"""Please download {url} to path {out_dir}""")
    return SimulationIterator(
        [p for p in list(map(lambda x: path.join(out_dir, x), os.listdir(out_dir))) if p.endswith('.csv')], test=test)
