import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist, squareform, pdist
from sklearn.model_selection import cross_val_predict
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler


def propensity_score_l1_based(clf, x, t, y):
    """
    Using a given clf, calculate propensity scores and returns ATE, ATC and ATT
    based on L1 distance matching

    Parameters
    ----------
    clf : propensity score model, with ability to predict_proba
    x: covariates
    t: binary treatment
    y: response

    Returns
    -------
    propensity_scores, ret
    Where ret is a dictionary with ATE, ATC and ATT values

    """
    y_0 = y.loc[t == 0]
    y_1 = y.loc[t == 1]
    # locs for numpy arrays
    loc_0 = y_0.index.values - 1
    loc_1 = y_1.index.values - 1

    propensity_scores = cross_val_predict(clf, x, t, cv=3, method='predict_proba')[:, 1]
    propensity_scores_0 = propensity_scores[loc_0]
    propensity_scores_1 = propensity_scores[loc_1]

    # get minimum distance indices for control and treated group
    propensity_0_min_dist_ind = cdist(propensity_scores_1.reshape(-1, 1), propensity_scores_0.reshape(-1, 1),
                                      'minkowski', p=1).argmin(axis=1)
    propensity_1_min_dist_ind = cdist(propensity_scores_0.reshape(-1, 1), propensity_scores_1.reshape(-1, 1),
                                      'minkowski', p=1).argmin(axis=1)

    # matched Ys
    y_0_matched = y_0.values[propensity_0_min_dist_ind]
    y_1_matched = y_1.values[propensity_1_min_dist_ind]

    # calculate ATT, ATC, ATE
    ret = {'ATT': np.mean(y_1.values - y_0_matched),
           'ATC': np.mean(y_1_matched - y_0.values)}
    ret['ATE'] = (ret['ATT'] * len(y_1) + ret['ATC'] * len(y_0)) / len(y)

    return propensity_scores, ret


def get_matching_pairs(treated_df, non_treated_df, scaler=True):
    """
    from https://stats.stackexchange.com/questions/206832/matched-pairs-in-python-propensity-score-matching
    """
    treated_x = treated_df.values
    non_treated_x = non_treated_df.values
    if scaler:
        scaler = StandardScaler()
    if scaler:
        scaler.fit(treated_x)
        treated_x = scaler.transform(treated_x)
        non_treated_x = scaler.transform(non_treated_x)

    nbrs = NearestNeighbors(n_neighbors=1, algorithm='ball_tree').fit(non_treated_x)
    distances, indices = nbrs.kneighbors(treated_x)
    indices = indices.reshape(indices.shape[0])
    matched = non_treated_df.iloc[indices]
    return matched


def nearest_neighbors_based(x, t, y):
    """
    Estimate ATE, ATC and ATT based on nearest neighbor matching.

    Parameters
    ----------
    x: covariates
    t: binary treatment
    y: response

    Returns
    -------
    ATE

    """
    x_0 = x.loc[t == 0]
    y_0 = y.loc[t == 0]
    x_1 = x.loc[t == 1]
    y_1 = y.loc[t == 1]

    ATT = np.mean(y_1.values - y_0[get_matching_pairs(x_1, x_0).index].values)
    ATC = np.mean(y_1[get_matching_pairs(x_0, x_1).index].values - y_0.values)
    ATE = (ATT * len(y_1) + ATC * len(y_0)) / len(y)

    return {'ATT': ATT, 'ATC': ATC, 'ATE': ATE}


def mahanabolis_matching_based(x, t, y):
    """
    Estimate ATE, ATC and ATT using Mahanabolis based matching.

    Parameters
    ----------
    x: covariates
    t: binary treatment
    y: response

    Returns
    -------
    ATE

    """
    x_0 = x.loc[t == 0]
    y_0 = y.loc[t == 0]
    x_1 = x.loc[t == 1]
    y_1 = y.loc[t == 1]

    # create distance matrix
    dist_mat = pd.DataFrame(data=squareform(pdist(x.values, 'mahalanobis')), index=x.index)
    # mask with inf
    dist_mat.loc[x_0.index, x_0.index] = np.inf
    dist_mat.loc[x_1.index, x_1.index] = np.inf

    # compute ATT, ATC, ATE
    ATT = np.mean(y_1[np.argmin(dist_mat.loc[x_0.index, :].values, axis=1)] - y_0.values)
    ATC = np.mean(y_1.values - y_0[np.argmin(dist_mat.loc[x_1.index, :].values, axis=1)])
    ATE = (ATT * len(y_1) + ATC * len(y_0)) / len(y)

    return {'ATT': ATT, 'ATC': ATC, 'ATE': ATE}
