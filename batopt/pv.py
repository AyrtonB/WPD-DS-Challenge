# AUTOGENERATED! DO NOT EDIT! File to edit: nbs/07-pv-forecast.ipynb (unless otherwise specified).

__all__ = ['construct_df_charge_features', 'prepare_training_input_data', 'plot_random_day',
           'generate_kfold_preds_weeks', 'generate_kfold_charge_preds', 'predict_charge', 'get_train_test_arr',
           'get_train_test_Xy', 'predict_charge', 'fit_and_save_pv_model', 'prepare_test_feature_data',
           'optimise_test_charge_profile']

# Cell
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from moepy.lowess import quantile_model

from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.metrics import make_scorer, r2_score, mean_absolute_error, mean_squared_error
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import GroupKFold


from mlxtend.feature_selection import SequentialFeatureSelector as SFS

from skopt.plots import plot_objective
from skopt.space import Real, Categorical, Integer

from batopt import clean, discharge, utils, charge

import FEAutils as hlp

from ipypb import track

# Cell
def construct_df_charge_features(df, dt_rng=None):
    if dt_rng is None:
        dt_rng = pd.date_range(df.index.min(), df.index.max(), freq='30T')

    df_features = pd.DataFrame(index=dt_rng)

    # Adding temperature data
    temp_loc_cols = df.columns[df.columns.str.contains('temp_location')]
    df_features.loc[df.index, temp_loc_cols] = df[temp_loc_cols].copy()
    df_features = df_features.ffill(limit=1)

    # Adding solar irradiance data
    solar_loc_cols = df.columns[df.columns.str.contains('solar_location')]
    df_features.loc[df.index, solar_loc_cols] = df[solar_loc_cols].copy()
    df_features = df_features.ffill(limit=1)

    # Adding avg solar from previous week
    df_features['pv_7d_lag'] = df['pv_power_mw'].rolling(48*7).mean().shift(48*7)

    # Adding datetime features
    dts = df_features.index

    df_features['hour'] = dts.hour + dts.minute/60
    df_features['doy'] = dts.dayofyear

    # Removing some extraneous features - found not be particularly useful
    cols = [c for c in df_features.columns if 'solar_location4' not in c and 'solar_location1' not in c]
    df_features = df_features.filter(cols)

    # Removing NaN values
    df_features = df_features.dropna()

    return df_features

def prepare_training_input_data(intermediate_data_dir, start_hour=5):
    # Loading input data
    df = clean.combine_training_datasets(intermediate_data_dir).interpolate(limit=1)
    df_features = construct_df_charge_features(df)

    # Filtering for overlapping feature and target data
    dt_idx = pd.date_range(df_features.index.min(), df['pv_power_mw'].dropna().index.max()-pd.Timedelta(minutes=30), freq='30T')

    s_pv = df.loc[dt_idx, 'pv_power_mw']
    df_features = df_features.loc[dt_idx]

    # Filtering for evening datetimes
    charging_datetimes = charge.extract_charging_datetimes(df_features, start_hour=start_hour)

    X = df_features.loc[charging_datetimes]
    y = s_pv.loc[charging_datetimes]

    return X, y

# Cell
def plot_random_day(df_pred, ax=None):
    """
    View predicted and observed PV profiles
    """
    if ax is None:
        ax = plt.gca()

    random_day_idx = pd.to_datetime(np.random.choice(df_pred.index.date))
    df_random_day = df_pred[df_pred.index.date==random_day_idx]

    df_random_day['true'].plot(ax=ax)
    df_random_day['pred'].plot(ax=ax)

    return ax

# Cell
def generate_kfold_preds_weeks(X, y, model, groups, kfold_kwargs, index=None):
    """
    Generate kfold preds, grouping by week
    """

    group_kfold = GroupKFold(**kfold_kwargs)

    df_pred = pd.DataFrame(columns=['pred', 'true'], index=np.arange(X.shape[0]))

    for train_index, test_index in group_kfold.split(X, y, groups):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        model.fit(X_train, y_train)

        df_pred.loc[test_index, 'true'] = y_test
        df_pred.loc[test_index, 'pred'] = model.predict(X_test)

    df_pred.sort_index()

    if index is not None:
        assert len(index) == df_pred.shape[0], 'The passed index must be the same length as X and y'
        df_pred.index = index

    return df_pred

def generate_kfold_charge_preds(X, y, model, groups, kfold_kwargs={'n_splits': 5}):
    """
    Fit the PV forecasting model and calculate the optimal charge profile for predictions.
    """
    df_pred = generate_kfold_preds_weeks(X.values, y.values, model, groups, kfold_kwargs=kfold_kwargs, index=X.index)

    charge_pred = charge.construct_charge_s(df_pred.pred)
    charge_pred = charge.post_pred_charge_proc_func(charge_pred)

    return pd.DataFrame({'charge_pred': charge_pred,
                         'pv_actual': df_pred.true,
                         'pv_pred': df_pred.pred})

def predict_charge(X, model):
    """
    Given a fitted PV forecast model and feature array X, get the optimal charge profile.
    """
    pv_pred = pd.Series(model.predict(X), index=X.index)
    charge_pred = charge.construct_charge_s(pv_pred)
    charge_pred = charge.post_pred_charge_proc_func(charge_pred)
    return pd.Series(charge_pred, index=X.index)

# Cell
def get_train_test_arr(arr, start_of_test_period):
    train_arr = arr[:pd.to_datetime(start_of_test_period, utc=True)]
    test_arr = arr[pd.to_datetime(start_of_test_period, utc=True):]

    return train_arr, test_arr

def get_train_test_Xy(X, y, start_of_test_period):
    x_train, x_test = get_train_test_arr(X, start_of_test_period)
    y_train, y_test = get_train_test_arr(y, start_of_test_period)

    return x_train, x_test, y_train, y_test

# Cell
def predict_charge(X, model):
    """
    Given a fitted PV forecast model and feature array X, get the optimal charge profile.
    """
    pv_pred = pd.Series(model.predict(X), index=X.index)
    charge_pred = charge.construct_charge_s(pv_pred)
    charge_pred = charge.post_pred_charge_proc_func(charge_pred)

    return pd.Series(charge_pred, index=X.index)

# Cell
def fit_and_save_pv_model(X, y, pv_model_fp, model_class=LinearRegression, **model_params):
    model = model_class(**model_params)
    model.fit(X, y)

    with open(pv_model_fp, 'wb') as fp:
        joblib.dump(model, fp)

    return

# Cell
#exports
def prepare_test_feature_data(raw_data_dir, intermediate_data_dir, test_start_date=None, test_end_date=None, start_time='08:00', end_time='23:59'):
    # Loading input data
    df = clean.combine_training_datasets(intermediate_data_dir).interpolate(limit=1)
    df_features = construct_df_charge_features(df)

    # Loading default index (latest submission)
    if test_end_date is None or test_start_date is None:
        index = discharge.load_latest_submission_template(raw_data_dir).index
    else:
        index = df_features[test_start_date:test_end_date].index

    # Filtering feature data on submission datetimes
    df_features = df_features.loc[index].between_time(start_time, end_time)

    return df_features

def optimise_test_charge_profile(raw_data_dir, intermediate_data_dir, pv_model_fp, test_start_date=None, test_end_date=None, start_time='08:00', end_time='23:59'):
    df_features = prepare_test_feature_data(raw_data_dir, intermediate_data_dir, test_start_date=test_start_date, test_end_date=test_end_date, start_time=start_time, end_time=end_time)
    charging_datetimes = charge.extract_charging_datetimes(df_features)
    X_test = df_features.loc[charging_datetimes]

    model = discharge.load_trained_model(pv_model_fp)
    charge_profile = predict_charge(X_test, model)

    s_charge_profile = pd.Series(charge_profile, index=charging_datetimes)
    s_charge_profile = s_charge_profile.reindex(df_features.index).fillna(0)
    s_charge_profile = charge.post_pred_charge_proc_func(s_charge_profile)

    assert charge.charge_is_valid(s_charge_profile), "Charging profile is invalid"

    return s_charge_profile