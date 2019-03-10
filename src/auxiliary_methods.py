import pandas as pd
from heapq import nsmallest

import numpy as np
import scipy.spatial.distance as distance
import shap
import sklearn


def calc_distance(x, func="euclidean"):
    """
    Computes distances
    :param x: Matrix of NxM where N is number of samples and M is number of features.
    :param func: Distance metric: euclidean, mahalanobis
    :return: Distance metric of NxN
    """
    if x.shape[1] == 1:
        x = x.reshape(x.shape[0], 1)
    return distance.squareform(distance.pdist(x, metric=func))


def get_trained_model(x, y, base_estimator='sklearn.ensemble.gradient_boosting.GradientBoostingRegressor'):
    """
    Trains the model and return it
    :param x: matrix
    :param y: predicted
    :param base_estimator: String with the name of the base estimator
        (default value is 'sklearn.ensemble.gradient_boosting.GradientBoostingRegressor')
    :return: trained model
    """
    __import__(base_estimator.split('.')[0])
    m = eval(base_estimator + '()')
    m.fit(x, y)
    return m


def get_propensity_score(x, t):
    """

    :param x:
    :param t:
    :return:
    """
    m = get_trained_model(x, t, 'sklearn.ensemble.gradient_boosting.GradientBoostingClassifier')
    return m.predict_proba(x)


def get_inverse_propensity_weighing(x, y):
    """

    :param x:
    :param y:
    :return:
    """
    propensity = get_propensity_score(x, y)
    return propensity


def calc_inv_propensity_score_weighting(df):
    """
    Inverse Propensity Score Weighting (IPW) for Average Treatment Affect for Treated (ATT)
    Aim: to weight the treated and controlled observations to make them representative of the population of interest.

    ATT = E[Y1−Y0|T=1] = E_X (E(Y1−Y0|X,T=1)) = E_X ( E(Y1|X,T=1) − E(Y0|X,T=1) )

    Under SITA: E(Y0|X,T=1) = E(Y0|X,T=0)

    Therefore: ATT = E11_X ( E(Y1|X,T=1) − E(Y0|X,T=0) )

    Using propensity score e(X) we obtain:
    ATT = E_e(X) ( E(Y1|e(X),T=1) − E(Y0|e(X),T=0) )

    The propensity score and inverse probability of treatment weighting in average treatment effect in the treated (ATT)
    See:
    https://onlinelibrary.wiley.com/doi/full/10.1002/sim
    ##IPW_ATT = 1/n * (A - (np.dot(B, C)).sum()).6607
    https://www.ncbi.nlm.nih.gov/pubmed/28436047


    # Assumptions:
    Strongly ignorable treatment assumption (SITA):
    the potential outcomes (Y0,Y1) are independent of the treatment selection given the observed covariates X
    :return:
    """

    # Number of treated subjects in the dataset
    n1 = df['T'].sum()

    # The weight for a treated subject is taken as 1
    # The sum of outcomes (sum_i{Yi}) for all subjects received treatment (T=1)
    A = df.loc[df['T'] == 1, ['Y']].sum().values

    # List of outcomes Y for subjects that didn't receive treatment (T=0)
    B = df.loc[df['T'] == 0, ['Y']].values.reshape(-1)

    propensity_scores = get_propensity_score(df.drop(columns=['T', 'Y']), df['T'])

    # The weight for a control subject is defined as e(X)/1−e(X)
    C = (propensity_scores / (1 - propensity_scores))[(df['T'] == 0).values]

    # The IPW estimator for ATT
    IPW_ATT = (1 / n1 * A - ((np.dot(B, C)).sum() / C.sum()))

    return IPW_ATT[0]


def calc_matching(df, dist_func='euclidean', k=1):
    """
    Computes the Average Treatment Effect on the treated through matching.
    :param df: DataFrame with the following columns:
        1. Features to be considered when comparing two individuals.
        2. Treatment (boolean).
        3. Outcome Y.
    :param dist_func: Distance metric for evaluating the distances between individuals.
        Will be used as a scipy.distance metric.
    :param k: Number of neighbors to consider.
    :return: An Average Treatment Effect scalar.
    """
    x = df.drop(['T', 'Y'], axis=1)

    d = calc_distance(x.values, dist_func)

    ATTs = []

    # Get the treated (T=1) and controls (T=0)
    idx_treated = [i for i, v in enumerate((df['T'] == 1).values) if v]
    idx_untreated = [i for i, v in enumerate((df['T'] == 0).values) if v]

    # For every subject with T=1, calculate the average Y0 of its neighbors
    for index in idx_treated:
        # Get outcome of current treated
        current_outcome = df.iloc[index]['Y']

        # Finding the k-closest neighbors in T=0 based on their propensity score
        neighbors_indices = nsmallest(k, idx_untreated, key=lambda untreated_index: d[index, untreated_index])

        # Get the outcome of neighbors
        neighbors_outcome = [df.iloc[idx]['Y'] for idx in neighbors_indices]

        # Average the outcome of neighbors, and then subtract with Y1 outcome (current_outcome)
        avg_outcome = current_outcome - np.mean(neighbors_outcome)

        ATTs.append(avg_outcome)

    # ATT matching by propensity
    ATT = sum(ATTs) / len(ATTs)

    return ATT


def get_shap_values(model, x):
    """
    Returns a DataFrame of shap values.
    :param model: A trained model.
    :param x: A DataFrame of N instances x M features that can be given as input to the model.
    :return: A matrix of NxM where i,j is the Shap value of instance i's feature j.
    """
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(x)
    shap_values_df = pd.DataFrame(data=shap_values, columns=x.columns)
    return shap_values_df


def calc_model_t_shap_matching(df, dist_func='euclidean', k=[1]):
    """
    Compute the Average Treatment Effect by matching based on Shap values of the model predicting T.
    :param df: A DataFrame of individuals.
    :param dist_func: A distance metric for Scipy's pdist function.
    :param k: Number of neighbors to consider. This value comes as an array so that all possible values are considered.
    :return: Average Treatment Effect.
    """
    x = df.drop(columns=['Y', 'T'])
    t = df['T']
    model = get_trained_model(x, t)
    shap_values = get_shap_values(model, x)
    features_and_outcomes = shap_values.copy()
    features_and_outcomes['T'] = t
    features_and_outcomes['Y'] = df['Y']
    ATTs = []
    for current_k in k:
        ATT = calc_matching(features_and_outcomes, dist_func=dist_func, k=current_k)
        ATTs.append(ATT)
    return ATTs


def calc_model_y_shap_matching(df, dist_func='euclidean', k=[1]):
    """
    Compute the Average Treatment Effect by matching based on Shap values of the model predicting Y.
    :param df: A DataFrame of individuals.
    :param dist_func: A distance metric for Scipy's pdist function.
    :param k: Number of neighbors to consider. This value comes as an array so that all possible values are considered.
    :return: Average Treatment Effect.
    """
    x = df.drop(columns=['Y'])
    t = df['T']
    y = df['Y']
    try:
        model = get_trained_model(x, y)
    except:
        print("Failed running: model = get_trained_model(x, y)")
        return {'x':x, 'y':y, 't':t}
    shap_values = get_shap_values(model, x)
    features_and_outcomes = shap_values.copy()
    features_and_outcomes.drop(columns=['T'], inplace=True)
    features_and_outcomes['T'] = t
    features_and_outcomes['Y'] = y
    ATTs = []
    for current_k in k:
        ATT = calc_matching(features_and_outcomes, dist_func=dist_func, k=current_k)
        ATTs.append(ATT)
    return ATTs


def calc_model_y_and_model_t_shap_matching(df, combining_method='ratio', dist_func='euclidean', k=[1]):
    """
    Compute the Average Treatment Effect by matching based on Shap value of both Y predicting and T predicting models.
    :param df: A DataFrame of individuals.
    :param combining_method: The method of combining shap values of both models. Available methods:
        - 'ratio': ratio of Y-based Shap values divided by T-based Shap values.
    :param dist_func: A distance metric for Scipy's pdist function.
    :param k: Number of neighbors to consider. This value comes as an array so that all possible values are considered.
    :return: Average Treatment Effect.
    """
    x = df.drop(columns=['Y'])
    t = df['T']
    y = df['Y']

    model_t = get_trained_model(x.drop(columns=['T']), t)
    shap_t_values = get_shap_values(model_t, x.drop(columns=['T']))

    model_y = get_trained_model(x, y)
    shap_y_values = get_shap_values(model_y, x)

    if combining_method == 'ratio':
        combined_values = np.true_divide(shap_y_values.drop(columns=['T']), shap_t_values)
    else:
        raise IOError(
            f"combining_method is {combining_method} which is not implemented in "
            f"calc_model_y_and_model_t_shap_matching.")

    combined_values['T'] = t
    combined_values['Y'] = y
    ATTs = []
    for current_k in k:
        ATT = calc_matching(combined_values, dist_func=dist_func, k=current_k)
        ATTs.append(ATT)
    return ATTs