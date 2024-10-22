# AUTOGENERATED! DO NOT EDIT! File to edit: nbs/06-tuning.ipynb (unless otherwise specified).

__all__ = ['get_train_test_arr', 'get_train_test_Xy', 'calculate_pct_peak_reduction_s', 'calculate_emissions_factor_s',
           'calculate_score_s', 'score_charge', 'score_discharge', 'max_charge_score', 'calculate_score_s',
           'evaluate_submission', 'feature_selection']

# Cell
import json
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from mlxtend.feature_selection import SequentialFeatureSelector as SFS
from skopt.plots import plot_objective
from skopt.space import Real, Categorical, Integer

from batopt import clean, discharge, charge, pv, utils

import os
from ipypb import track

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
def calculate_pct_peak_reduction_s(discharge_y_pred, s_demand):
    s_demand_test = s_demand.loc[discharge_y_pred.index]

    s_old_peaks = s_demand_test.groupby(s_demand_test.index.date).max()
    s_new_peaks = (s_demand_test+discharge_y_pred).groupby(s_demand_test.index.date).max()

    s_pct_peak_reduction = 100*(s_old_peaks - s_new_peaks)/s_new_peaks
    s_pct_peak_reduction.index = pd.to_datetime(s_pct_peak_reduction.index)

    return s_pct_peak_reduction

# Cell
def calculate_emissions_factor_s(charge_y_pred, s_pv, solar_factor=3, grid_factor=1):
    s_solar_charge_pct = (charge_y_pred - s_pv.loc[charge_y_pred.index]).clip(0).groupby(charge_y_pred.index.date).sum()/charge_y_pred.groupby(charge_y_pred.index.date).sum()
    s_grid_charge_pct = 1 - s_solar_charge_pct

    s_emissions_factor = solar_factor*s_solar_charge_pct + grid_factor*s_grid_charge_pct
    s_emissions_factor.index = pd.to_datetime(s_emissions_factor.index)

    return s_emissions_factor

# Cell
def calculate_score_s(discharge_y_pred, charge_y_pred, s_demand, s_pv, solar_factor=3, grid_factor=1):
    s_pct_peak_reduction = calculate_pct_peak_reduction_s(discharge_y_pred, s_demand)
    s_emissions_factor = calculate_emissions_factor_s(charge_y_pred, s_pv, solar_factor=solar_factor, grid_factor=grid_factor)

    s_score = s_pct_peak_reduction*s_emissions_factor

    return s_score

# Cell
def score_charge(schedule, solar_profile, solar_factor=3, grid_factor=1):
    # The actual pv charge is the minimum of the scheduled charge and the actual solar availability
    actual_pv_charge = np.minimum(schedule.values, solar_profile.values)
    actual_pv_charge = pd.Series(actual_pv_charge, index=schedule.index)

    pct_pv_charge = actual_pv_charge.groupby(actual_pv_charge.index.date).sum() / schedule.groupby(schedule.index.date).sum()
    pct_grid_charge = 1 - pct_pv_charge

    score = (solar_factor * pct_pv_charge) + (grid_factor * pct_grid_charge)

    return score

def score_discharge(schedule, demand):

    new_demand = schedule + demand
    old_demand = demand

    new_peaks = new_demand.groupby(new_demand.index.date).max()
    old_peaks = old_demand.groupby(old_demand.index.date).max()

    pct_reduction = 100*((old_peaks - new_peaks)/ old_peaks)

    return pct_reduction

def max_charge_score(solar_profile, solar_factor=3, grid_factor=1, capacity=6, time_unit=0.5):
    pv_potential = solar_profile.groupby(solar_profile.index.date).sum().clip(0, capacity/time_unit)
    pct_pv_charge = pv_potential / (capacity/time_unit)
    pct_grid_charge = 1 - pct_pv_charge

    score = (solar_factor * pct_pv_charge) + (grid_factor * pct_grid_charge)

    return score


def calculate_score_s(discharge_y_pred, charge_y_pred, s_demand, s_pv, solar_factor=3, grid_factor=1):

    charge_score = score_charge(charge_y_pred, s_pv, solar_factor, grid_factor)
    discharge_score = score_discharge(discharge_y_pred, s_demand)

    s_score = discharge_score*charge_score

    return s_score, charge_score, discharge_score

def evaluate_submission(submission, intermediate_data_dir):
    if isinstance(submission, str):
        df_solution = pd.read_csv(submission)
        df_solution = df_solution.set_index(pd.to_datetime(df_solution.datetime, utc=True))
    else:
        assert isinstance(submission, pd.DataFrame), '`submission` must either be a valid submission dataframe or a filepath to the submission'
        df_solution = submission

    df_real = clean.combine_training_datasets(intermediate_data_dir)
    df_real = df_real[df_real.index.isin(df_solution.index)]

    df_solution_charge = df_solution.between_time('00:00', '15:00')
    df_solution_discharge = df_solution.between_time('15:30', '20:30')

    df_real_charge = df_real.between_time('00:00', '15:00')
    df_real_discharge = df_real.between_time('15:30', '20:30')

    total_score, charge_score, discharge_score = calculate_score_s(df_solution_discharge.charge_MW, df_solution_charge.charge_MW, df_real_discharge.demand_MW, df_real_charge.pv_power_mw)

    df_results = pd.DataFrame({
        'total_score': total_score,
        'charge_score': charge_score,
        'discharge_score': discharge_score,
        'max_charge_score': max_charge_score(df_real_charge.pv_power_mw)
    })

    return df_results

# Cell
def feature_selection(x_train, y_train, groups=None, model=RandomForestRegressor(), min_num_features=1, max_num_features=None, **sfs_kwargs):
    if max_num_features is None:
        max_num_features = 1 + x_train.shape[1]

    result_features = dict()
    result_scores = dict()

    for num_features in track(range(min_num_features, max_num_features)):
        sfs = SFS(
            model,
            k_features=num_features,
            **sfs_kwargs
        )

        sfs.fit(x_train, y_train, groups=groups)

        result_features[num_features] = sfs.k_feature_names_
        result_scores[num_features] = sfs.k_score_

    return result_features, result_scores