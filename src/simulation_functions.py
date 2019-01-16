import numpy as np


def kang_shafer_data(size=(1,) ,random_seed=0, X_observed=True):
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
    X1 = np.random.normal(0,1,size=size)
    X2 = np.random.normal(0,1,size=size)
    X3 = np.random.normal(0,1,size=size)
    X4 = np.random.normal(0,1,size=size)

    true_propensity = 1/(1+np.exp(X1-0.5*X2+0.25*X3+0.1*X4))
    A = (true_propensity>=0.5).astype(int)
    # print(X1, X2, X3, X4, true_propensity, A)
    
    E_Y = 210 + A + 27.4*X1 + 13.78*X2 + 13.7*X3 + 13.7*X4

    Y_true = np.random.normal(E_Y,1)
    
    if X_observed:
        return X1, X2, X3, X4, A, Y_true
    else: # Xs are "not observed" , but a nonlinear transformation of them is
        X1_t = np.exp(X1/2)
        X2_t = (X2/(1+np.exp(X1))) + 10
        X3_t = (((X1*X3)/25) + 0.6)**3
        X4_t = (X2 + X4 + 20)**2
        return X1_t, X2_t, X3_t, X4_t, A, Y_true