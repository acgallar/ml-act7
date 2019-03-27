#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, median_absolute_error, r2_score
from sklearn.model_selection import train_test_split, GridSearchCV


def identify_object_type_variables(df):
    """Given a pd.DataFrame object, return a list with those variables that are object type
    """
    # for each column, preserve name if it's an object dtype, empty string otherwise
    tmp_vartype = [df[i].name if df[i].dtype == 'object' else '' for i in df.columns]
    # preserve names
    tmp_vartype = list(filter(lambda x: x != '', tmp_vartype))
    # return list
    return tmp_vartype


def propagate_dummies(df, object_type_list):
    """
    Given a pd.DataFrame and a list contaning columns, impute $k-1$ categories in dummies.
    Returns a refactorized dataframe
    """
    # set an empty list holder
    tmp_hold_dummies = []
    # for each object dtype variable
    for  i in object_type_list:
        # append to the list holder
        tmp_hold_dummies.append(
            # generate k-1 dummies (disregarding reference assignment)
            pd.get_dummies(df[i],
                           # set prefix name
                           prefix=df[i].name,
                           # set separator
                           prefix_sep='_',
                           drop_first=True,
                           dummy_na=False)
        )

    # concatenate dummies
    tmp_df = pd.concat(
        [df, pd.concat(
            tmp_hold_dummies, axis=1
        )], axis=1
    )
    # remove original variables
    tmp_df = tmp_df.drop(columns=object_type_list)
    # return
    return tmp_df

def plot_feature_importance(fit_model, feat_names):
    """
    Plot relative importance of a feature subset given a fitted model.
    """
    # infer feature importance score
    tmp_importance = fit_model.feature_importances_
    # sort features
    sort_importance = np.argsort(tmp_importance)[::-1]
    # associate feat_names with its relative importance
    names = [feat_names[i] for i in sort_importance]
    # plot
    plt.barh(
        # given range and features
        range(len(feat_names)), tmp_importance[sort_importance])
    
    # add axis labels identifying attribute name
    plt.yticks(range(len(feat_names)), 
               names, rotation=0)


def infer_k_features(df, model, feat_names, k_feats=10):
    """TODO: Docstring for infer_k_features.

    :arg1: TODO
    :returns: TODO

    """
    # preserve temporary copy
    tmp_df = df.copy()
    # infer feature importance score
    tmp_importance = model.feature_importances_
    # sort features
    sort_importance = np.argsort(tmp_importance)[::-1]
    # associate feat names with its relative importance
    names = [feat_names[i] for i in sort_importance]
    #mungle into a dataframe
    tmp_attr = pd.DataFrame(
        {'name': names,
         'score': tmp_importance[sort_importance]}
    )
    # restrict dataframe to k attributes
    tmp_attr = tmp_attr[:k_feats]['name']

    # filter attributes
    tmp_df = tmp_df[
        tmp_df.columns[
            tmp_df.columns.isin(tmp_attr)
        ]
    ]

    return tmp_df


