# Tuning



```python
#exports
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
```

```python
import FEAutils as hlp
import matplotlib.pyplot as plt

from IPython.display import JSON
```

<br>

### User Inputs

```python
intermediate_data_dir = '../data/intermediate'
raw_data_dir = '../data/raw'
cache_data_dir = '../data/nb-cache'
```

<br>

### Preparing Data

First we'll load in the target and feature data for both the charging and discharging models

```python
charge_x, charge_y = pv.prepare_training_input_data(intermediate_data_dir)
discharge_x, discharge_y = discharge.prepare_training_input_data(intermediate_data_dir)

charge_x.head()
```

```python
s_demand = clean.load_training_dataset(intermediate_data_dir, 'demand')['demand_MW']

s_demand.head()
```

```python
s_pv = clean.load_training_dataset(intermediate_data_dir, 'pv')['pv_power_mw']

s_pv.head()
```

```python
#exports
def get_train_test_arr(arr, start_of_test_period): 
    train_arr = arr[:pd.to_datetime(start_of_test_period, utc=True)]
    test_arr = arr[pd.to_datetime(start_of_test_period, utc=True):]
    
    return train_arr, test_arr

def get_train_test_Xy(X, y, start_of_test_period): 
    x_train, x_test = get_train_test_arr(X, start_of_test_period)
    y_train, y_test = get_train_test_arr(y, start_of_test_period)
    
    return x_train, x_test, y_train, y_test
```

```python
start_of_test_period = '2020-06-15'

charge_x_train, charge_x_test, charge_y_train, charge_y_test = pv.get_train_test_Xy(charge_x, charge_y, start_of_test_period)
discharge_x_train, discharge_x_test, discharge_y_train, discharge_y_test = pv.get_train_test_Xy(discharge_x, discharge_y, start_of_test_period)
```

<br>

### Evaluation Metrics

We want to evaluate each of our models based on their contribution to the final scoring value, to do this we'll first create some predictions for our discharge model.

```python
discharge_rf = RandomForestRegressor()

discharge_rf.fit(discharge_x_train, discharge_y_train)
discharge_y_pred = pd.Series(discharge_rf.predict(discharge_x_test), index=discharge_x_test.index)

discharge_y_pred.plot()
```

<br>

We'll then create a time-series of the percentage peak reduction for each day

```python
#exports
def calculate_pct_peak_reduction_s(discharge_y_pred, s_demand):
    s_demand_test = s_demand.loc[discharge_y_pred.index]

    s_old_peaks = s_demand_test.groupby(s_demand_test.index.date).max()
    s_new_peaks = (s_demand_test+discharge_y_pred).groupby(s_demand_test.index.date).max()

    s_pct_peak_reduction = 100*(s_old_peaks - s_new_peaks)/s_new_peaks
    s_pct_peak_reduction.index = pd.to_datetime(s_pct_peak_reduction.index)

    return s_pct_peak_reduction
```

```python
s_pct_peak_reduction = calculate_pct_peak_reduction_s(discharge_y_pred, s_demand)

print(f'The average peak reduction was {s_pct_peak_reduction.mean():.2f}%')

s_pct_peak_reduction.plot()
```

<br>

We'll then repeat this with the charging model

```python
charge_rf = RandomForestRegressor()

charge_rf.fit(charge_x_train, charge_y_train)
charge_y_pred = pd.Series(charge_rf.predict(charge_x_test), index=charge_x_test.index)

charge_y_pred.plot()
```

<br>

For which we'll calculate the emissions factor series

```python
#exports
def calculate_emissions_factor_s(charge_y_pred, s_pv, solar_factor=3, grid_factor=1):
    s_solar_charge_pct = (charge_y_pred - s_pv.loc[charge_y_pred.index]).clip(0).groupby(charge_y_pred.index.date).sum()/charge_y_pred.groupby(charge_y_pred.index.date).sum()
    s_grid_charge_pct = 1 - s_solar_charge_pct

    s_emissions_factor = solar_factor*s_solar_charge_pct + grid_factor*s_grid_charge_pct
    s_emissions_factor.index = pd.to_datetime(s_emissions_factor.index)

    return s_emissions_factor
```

```python
s_emissions_factor = calculate_emissions_factor_s(charge_y_pred, s_pv)

s_emissions_factor.plot()
```

<br>

We can then combine these two steps to determine our final score for each day

```python
s_score = calculate_score_s(discharge_y_pred, charge_y_pred, s_demand, s_pv)

print(f'The average score was: {s_score.mean():.2f}')

s_score.plot()
```

<br>

For the charging we can also look at how much was sourced from PV relative to the potential maximum (capped at 6 MWh per day).

```python
solar_charge = np.minimum(charge_y_pred, s_pv.loc[charge_y_pred.index])
day_solar_charge = solar_charge.groupby(solar_charge.index.date).sum().clip(0,12)
day_solar_charge.index = pd.to_datetime(day_solar_charge.index)

solar_potential = np.clip(s_pv.loc[charge_y_pred.index], 0, 2.5)
day_solar_potential = solar_potential.groupby(solar_potential.index.date).sum().clip(0,12)
day_solar_potential.index = pd.to_datetime(day_solar_potential.index)

day_solar_charge.plot()
day_solar_potential.plot()
```

```python
pct_exploit = 100 * day_solar_charge/day_solar_potential
pct_exploit.plot()
plt.ylabel('% exploited')
```

```python
#exports
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
```

```python
submission_fp = '../data/output/ESAIL_set1.csv'

df_results = evaluate_submission(submission_fp, intermediate_data_dir)

df_results
```

<br>

We can then calculate our average score over this period

```python
df_results['total_score'].mean()
```

<br>

### Discharge Model Tuning

We'll begin by carrying out some feature selection

```python
#exports
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
```

```python
peak_reduction_scorer = discharge.construct_peak_reduction_calculator(s_demand=s_demand.loc[discharge_x_train.index], scorer=True)
week_groups = discharge_x_train.index.year + discharge_x_train.index.isocalendar().week/52

rerun_feature_selection = False
feature_selection_filename = f'feature_selection.json'

if (rerun_feature_selection == True) or (feature_selection_filename not in os.listdir(cache_data_dir)):
    result_features, result_scores = feature_selection(discharge_x_train, discharge_y_train, groups=week_groups, n_jobs=-1)
    
    with open(f'{cache_data_dir}/{feature_selection_filename}', 'w') as fp:
        json.dump(dict(zip(['features', 'scores'], [result_features, result_scores])), fp)
        
else:
    with open(f'{cache_data_dir}/{feature_selection_filename}', 'r') as fp:
        results = json.load(fp)
        
    result_features, result_scores = results['features'], results['scores']
```

<br>

We can visualise how the model accuracy changes with the number of features included

```python
pd.Series(result_scores).plot()
```

<br>

We'll also calculate the relative importance of each feature by counting how many times they were included in the optimal feature subset

```python
flatten_iterables = lambda iterable: [item for sublist in list(iterable) for item in sublist]

s_feature_importance = pd.Series(flatten_iterables(result_features.values())).value_counts().divide(len(result_features))

s_feature_importance
```

<br>

We'll now do some hyper-parameter tuning using the `skopt` library

```python
features = s_feature_importance.index[:11]
evening_datetimes = discharge.extract_evening_datetimes(discharge_x_train)
week_groups = discharge_x_train.index.year + discharge_x_train.index.isocalendar().week/52
peak_reduction_scorer = discharge.construct_peak_reduction_calculator(s_demand=s_demand, scorer=True)

pipeline = Pipeline([
    # Add in oversampling of more recent/similar dates
    ('pandas_RF', utils.PandasRandomForestRegressor())
])

search_spaces = {
        'pandas_RF__min_samples_leaf': Integer(1, 20, 'uniform'),
        'pandas_RF__criterion': Categorical(['mse', 'mae']),
        'pandas_RF__n_estimators': Integer(50, 250, 'uniform'),
        'pandas_RF__max_features': Categorical(['auto', 'sqrt']),
        'pandas_RF__max_depth': Integer(10, 50, 'uniform'),
        'pandas_RF__min_samples_split': Integer(2, 10, 'uniform'),
        'pandas_RF__min_samples_leaf': Integer(1, 4, 'uniform'),
        'pandas_RF__bootstrap': Categorical([True, False])
}

opt = utils.BayesSearchCV(
    pipeline,
    search_spaces,
    n_iter=15,
    verbose=1,
    cv=8, # 8 works well for me as that's how many concurrent workers I can use
    scoring=peak_reduction_scorer,
    n_jobs=-1
)

fit_BayesSearchCV = False

if fit_BayesSearchCV == True:
    opt.fit(discharge_x_train[features], discharge_y_train, groups=evening_datetimes.date)

    print(f'Cross-validation score: {opt.best_score_:.2f}')
    print(f'Hold-out score: {opt.score(discharge_x_test[features], discharge_y_test):.2f}')
    print(f'\nBest params: \n{opt.best_params_}')
```

```python
# want to be saving model runs
# could include as part of a callback?
```

<br>

### Model Comparisons

Here we'll compare our discharge v pv-forecast modelling approaches

```python
index = pd.date_range('2019-03-02', '2019-03-09 23:30', freq='30T', tz='UTC')
```

```python
%%time

discharge_opt_model_fp = '../models/discharge_opt.sav'

X, y = discharge.prepare_training_input_data(intermediate_data_dir)
idxs_to_keep = sorted(list(set(X.index) - set(index)))
X, y = X.loc[idxs_to_keep], y.loc[idxs_to_keep]

discharge.fit_and_save_model(X, y, discharge_opt_model_fp)
s_discharge_profile = discharge.optimise_test_discharge_profile(raw_data_dir, intermediate_data_dir, discharge_opt_model_fp, index=index)

s_discharge_profile.plot()
```

```python
charge_opt_model_fp = '../models/charge_opt.sav'

X, y = charge.prepare_training_input_data(intermediate_data_dir, start_hour=5)
idxs_to_keep = sorted(list(set(X.index) - set(index)))
X, y = X.loc[idxs_to_keep], y.loc[idxs_to_keep]

charge.fit_and_save_charging_model(X, y, charge_opt_model_fp)
s_charge_profile = charge.optimise_test_charge_profile(raw_data_dir, intermediate_data_dir, charge_opt_model_fp, index=index)

s_charge_profile.plot()
```

<br>

We'll now create the charging profile using the PV forecast, in this instance we'll use a linear model for the solar forecast

```python
pv_model_fp = '../models/pv_model.sav'

X, y = pv.prepare_training_input_data(intermediate_data_dir, start_hour=5)
idxs_to_keep = sorted(list(set(X.index) - set(index)))
X, y = X.loc[idxs_to_keep], y.loc[idxs_to_keep]

pv.fit_and_save_pv_model(X, y, pv_model_fp)
s_charge_profile = pv.optimise_test_charge_profile(raw_data_dir, intermediate_data_dir, pv_model_fp, index=index)

s_charge_profile.plot()
```

<br>

In this example we repeat the same procedure using a random forest instead of linear model

```python
pv_model_fp = '../models/pv_model.sav'

X, y = pv.prepare_training_input_data(intermediate_data_dir, start_hour=5)
idxs_to_keep = sorted(list(set(X.index) - set(index)))
X, y = X.loc[idxs_to_keep], y.loc[idxs_to_keep]

pv.fit_and_save_pv_model(X, y, pv_model_fp, model_class=RandomForestRegressor)
s_charge_profile = pv.optimise_test_charge_profile(raw_data_dir, intermediate_data_dir, pv_model_fp, index=index)

s_charge_profile.plot()
```

```python
submission = (s_discharge_profile+s_charge_profile).to_frame(name='charge_MW')

df_results = evaluate_submission(submission, intermediate_data_dir)

df_results
```

```python
df_results['total_score'].mean()
```

```python
submission = (s_discharge_profile+s_charge_profile).to_frame(name='charge_MW')

df_results = evaluate_submission(submission, intermediate_data_dir)

df_results['total_score'].mean()
```

<br>

Finally we'll export the relevant code to our `batopt` module
