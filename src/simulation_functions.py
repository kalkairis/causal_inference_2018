import numpy as np


def kang_shafer_data(random_seed=0):
    """
        Generate Y_true from Kang-Schafer Data Generating Process
        
        Parameters
        ----------
        random_seed : numpy random seed. default=0
        
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
    X1 = np.random.normal(0,1)
    X2 = np.random.normal(0,1)
    X3 = np.random.normal(0,1)
    X4 = np.random.normal(0,1)

    true_propensity = 1/(1+np.exp(X1-0.5*X2+0.25*X3+0.1*X4))
    A = int(true_propensity>=0.5)
    # print(X1, X2, X3, X4, true_propensity, A)
    
    E_Y = 210 + A + 27.4*X1 + 13.78*X2 + 13.7*X3 + 13.7*X4

    Y_true = np.random.normal(E_Y,1)
    
    return A, Y_true