# Pipeline



```python
#exports
import numpy as np
import pandas as pd

import os
from sklearn.ensemble import RandomForestRegressor

from dagster import execute_pipeline, pipeline, solid, Field
from batopt import clean, discharge, charge, constraints, pv
```

```python
import FEAutils as hlp
import matplotlib.pyplot as plt
```

<br>

### End-to-End

We're now going to combine these steps into a pipeline using dagster, first we'll create the individual components.

```python
@solid()
def load_data(_, raw_data_dir: str):
    loaded_data = dict()
    
    loaded_data['pv'] = clean.load_training_dataset(raw_data_dir, 'pv')
    loaded_data['demand'] = clean.load_training_dataset(raw_data_dir, 'demand')
    loaded_data['weather'] = clean.load_training_dataset(raw_data_dir, 'weather', dt_idx_freq='H')
    
    return loaded_data

@solid()
def clean_data(_, loaded_data, raw_data_dir: str, intermediate_data_dir: str):
    # Cleaning
    cleaned_data = dict()

    cleaned_data['pv'] = (loaded_data['pv']
                          .pipe(clean.pv_anomalies_to_nan)
                          .pipe(clean.interpolate_missing_panel_temps, loaded_data['weather'])
                          .pipe(clean.interpolate_missing_site_irradiance, loaded_data['weather'])
                          .pipe(clean.interpolate_missing_site_power)
                         )
    cleaned_data['weather'] = clean.interpolate_missing_weather_solar(loaded_data['pv'], loaded_data['weather'])
    cleaned_data['weather'] = clean.interpolate_missing_temps(cleaned_data['weather'], 'temp_location4')
    
    cleaned_data['demand'] = loaded_data['demand']
    
    # Saving
    if os.path.exists(intermediate_data_dir) == False:
        os.mkdir(intermediate_data_dir)
        
    set_num = clean.identify_latest_set_num(raw_data_dir)
    
    cleaned_data['pv'].to_csv(f'{intermediate_data_dir}/pv_set{set_num}.csv')
    cleaned_data['demand'].to_csv(f'{intermediate_data_dir}/demand_set{set_num}.csv')
    cleaned_data['weather'].to_csv(f'{intermediate_data_dir}/weather_set{set_num}.csv')
            
    return intermediate_data_dir

@solid()
def fit_and_save_pv_model(_, intermediate_data_dir: str, pv_model_fp: str, model_params: dict):
    X, y = pv.prepare_training_input_data(intermediate_data_dir)
    pv.fit_and_save_pv_model(X, y, pv_model_fp, model_class=RandomForestRegressor, **model_params)
    
    return True

@solid()
def fit_and_save_discharge_model(_, intermediate_data_dir: str, discharge_opt_model_fp: str, model_params: dict):
    X, y = discharge.prepare_training_input_data(intermediate_data_dir)
    discharge.fit_and_save_model(X, y, discharge_opt_model_fp, **model_params)
    
    return True

@solid()
def construct_battery_profile(_, charge_model_success: bool, discharge_model_success: bool, intermediate_data_dir: str, raw_data_dir: str, discharge_opt_model_fp: str, pv_model_fp: str, start_time: str):
    assert charge_model_success and discharge_model_success, 'Model training was unsuccessful'
    
    s_discharge_profile = discharge.optimise_test_discharge_profile(raw_data_dir, intermediate_data_dir, discharge_opt_model_fp)
    s_charge_profile = pv.optimise_test_charge_profile(raw_data_dir, intermediate_data_dir, pv_model_fp, start_time=start_time)
    
    s_battery_profile = (s_charge_profile + s_discharge_profile).fillna(0)
    s_battery_profile.name = 'charge_MW'
    
    return s_battery_profile

@solid()
def check_and_save_battery_profile(_, s_battery_profile, output_data_dir: str):
    # Check that solution meets battery constraints
    assert constraints.schedule_is_legal(s_battery_profile), 'Solution violates constraints'
    
    # Saving
    if os.path.exists(output_data_dir) == False:
        os.mkdir(output_data_dir)
        
    s_battery_profile.index = s_battery_profile.index.tz_convert('UTC').tz_convert(None)
    s_battery_profile.to_csv(f'{output_data_dir}/latest_submission.csv')
    
    return 
```

<br>

Then we'll combine them in a pipeline

```python
@pipeline
def end_to_end_pipeline(): 
    # loading and cleaning
    loaded_data = load_data()
    intermediate_data_dir = clean_data(loaded_data)
    
    # charging
    charge_model_success = fit_and_save_pv_model(intermediate_data_dir)
    
    # discharing
    discharge_model_success = fit_and_save_discharge_model(intermediate_data_dir)
    
    # combining and saving
    s_battery_profile = construct_battery_profile(charge_model_success, discharge_model_success, intermediate_data_dir)
    check_and_save_battery_profile(s_battery_profile)
```

<br>

Which we'll now run a test

```python
run_config = {
    'solids': {
        'load_data': {
            'inputs': {
                'raw_data_dir': '../data/raw',
            },
        },
        'clean_data': {
            'inputs': {
                'raw_data_dir': '../data/raw',
                'intermediate_data_dir': '../data/intermediate',
            },
        },
        'fit_and_save_discharge_model': {
            'inputs': {
                'discharge_opt_model_fp': '../models/discharge_opt.sav',
                'model_params': {
                    'criterion': 'mse',
                    'bootstrap': True,
                    'max_depth': 32,
                    'max_features': 'auto',
                    'min_samples_leaf': 1,
                    'min_samples_split': 4,
                    'n_estimators': 74                    
                }
            },
        },
        'fit_and_save_pv_model': {
            'inputs': {
                'pv_model_fp': '../models/pv_model.sav',
                'model_params': {
                    'bootstrap': True,
                    'criterion': 'mse', 
                    'max_depth': 5,
                    'max_features': 'sqrt',
                    'min_samples_leaf': 1,
                    'min_samples_split': 2,
                    'n_estimators': 150
                }
            },
        },
        'construct_battery_profile': {
            'inputs': {
                'raw_data_dir': '../data/raw',
                'discharge_opt_model_fp': '../models/discharge_opt.sav',
                'pv_model_fp': '../models/pv_model.sav',
                'start_time': '08:00',
            },
        },
        'check_and_save_battery_profile': {
            'inputs': {
                'output_data_dir': '../data/output'
            },
        },
    }
}

execute_pipeline(end_to_end_pipeline, run_config=run_config)
```

    [32m2021-03-19 01:22:42[0m - dagster - [34mDEBUG[0m - end_to_end_pipeline - 01f88437-2b8c-4581-9982-3f76523c2874 - 17436 - ENGINE_EVENT - Starting initialization of resources [asset_store].
    [32m2021-03-19 01:22:42[0m - dagster - [34mDEBUG[0m - end_to_end_pipeline - 01f88437-2b8c-4581-9982-3f76523c2874 - 17436 - ENGINE_EVENT - Finished initialization of resources [asset_store].
    [32m2021-03-19 01:22:42[0m - dagster - [34mDEBUG[0m - end_to_end_pipeline - 01f88437-2b8c-4581-9982-3f76523c2874 - 17436 - PIPELINE_START - Started execution of pipeline "end_to_end_pipeline".
    [32m2021-03-19 01:22:42[0m - dagster - [34mDEBUG[0m - end_to_end_pipeline - 01f88437-2b8c-4581-9982-3f76523c2874 - 17436 - ENGINE_EVENT - Executing steps in process (pid: 17436)
    [32m2021-03-19 01:22:42[0m - dagster - [34mDEBUG[0m - end_to_end_pipeline - 01f88437-2b8c-4581-9982-3f76523c2874 - 17436 - load_data.compute - STEP_START - Started execution of step "load_data.compute".
    [32m2021-03-19 01:22:42[0m - dagster - [34mDEBUG[0m - end_to_end_pipeline - 01f88437-2b8c-4581-9982-3f76523c2874 - 17436 - load_data.compute - STEP_INPUT - Got input "raw_data_dir" of type "String". (Type check passed).
    [32m2021-03-19 01:22:42[0m - dagster - [34mDEBUG[0m - end_to_end_pipeline - 01f88437-2b8c-4581-9982-3f76523c2874 - 17436 - load_data.compute - STEP_OUTPUT - Yielded output "result" of type "Any". (Type check passed).
    [32m2021-03-19 01:22:42[0m - dagster - [34mDEBUG[0m - end_to_end_pipeline - 01f88437-2b8c-4581-9982-3f76523c2874 - 17436 - load_data.compute - OBJECT_STORE_OPERATION - Stored intermediate object for output result in memory object store using pickle.
    [32m2021-03-19 01:22:42[0m - dagster - [34mDEBUG[0m - end_to_end_pipeline - 01f88437-2b8c-4581-9982-3f76523c2874 - 17436 - load_data.compute - STEP_SUCCESS - Finished execution of step "load_data.compute" in 253ms.
    [32m2021-03-19 01:22:42[0m - dagster - [34mDEBUG[0m - end_to_end_pipeline - 01f88437-2b8c-4581-9982-3f76523c2874 - 17436 - clean_data.compute - STEP_START - Started execution of step "clean_data.compute".
    [32m2021-03-19 01:22:42[0m - dagster - [34mDEBUG[0m - end_to_end_pipeline - 01f88437-2b8c-4581-9982-3f76523c2874 - 17436 - clean_data.compute - OBJECT_STORE_OPERATION - Retrieved intermediate object for input loaded_data in memory object store using pickle.
    [32m2021-03-19 01:22:42[0m - dagster - [34mDEBUG[0m - end_to_end_pipeline - 01f88437-2b8c-4581-9982-3f76523c2874 - 17436 - clean_data.compute - STEP_INPUT - Got input "loaded_data" of type "Any". (Type check passed).
    [32m2021-03-19 01:22:42[0m - dagster - [34mDEBUG[0m - end_to_end_pipeline - 01f88437-2b8c-4581-9982-3f76523c2874 - 17436 - clean_data.compute - STEP_INPUT - Got input "raw_data_dir" of type "String". (Type check passed).
    [32m2021-03-19 01:22:42[0m - dagster - [34mDEBUG[0m - end_to_end_pipeline - 01f88437-2b8c-4581-9982-3f76523c2874 - 17436 - clean_data.compute - STEP_INPUT - Got input "intermediate_data_dir" of type "String". (Type check passed).
    [32m2021-03-19 01:24:43[0m - dagster - [34mDEBUG[0m - end_to_end_pipeline - 01f88437-2b8c-4581-9982-3f76523c2874 - 17436 - clean_data.compute - STEP_OUTPUT - Yielded output "result" of type "Any". (Type check passed).
    [32m2021-03-19 01:24:43[0m - dagster - [34mDEBUG[0m - end_to_end_pipeline - 01f88437-2b8c-4581-9982-3f76523c2874 - 17436 - clean_data.compute - OBJECT_STORE_OPERATION - Stored intermediate object for output result in memory object store using pickle.
    [32m2021-03-19 01:24:43[0m - dagster - [34mDEBUG[0m - end_to_end_pipeline - 01f88437-2b8c-4581-9982-3f76523c2874 - 17436 - clean_data.compute - STEP_SUCCESS - Finished execution of step "clean_data.compute" in 2m1s.
    [32m2021-03-19 01:24:43[0m - dagster - [34mDEBUG[0m - end_to_end_pipeline - 01f88437-2b8c-4581-9982-3f76523c2874 - 17436 - fit_and_save_discharge_model.compute - STEP_START - Started execution of step "fit_and_save_discharge_model.compute".
    [32m2021-03-19 01:24:43[0m - dagster - [34mDEBUG[0m - end_to_end_pipeline - 01f88437-2b8c-4581-9982-3f76523c2874 - 17436 - fit_and_save_discharge_model.compute - OBJECT_STORE_OPERATION - Retrieved intermediate object for input intermediate_data_dir in memory object store using pickle.
    [32m2021-03-19 01:24:43[0m - dagster - [34mDEBUG[0m - end_to_end_pipeline - 01f88437-2b8c-4581-9982-3f76523c2874 - 17436 - fit_and_save_discharge_model.compute - STEP_INPUT - Got input "intermediate_data_dir" of type "String". (Type check passed).
    [32m2021-03-19 01:24:43[0m - dagster - [34mDEBUG[0m - end_to_end_pipeline - 01f88437-2b8c-4581-9982-3f76523c2874 - 17436 - fit_and_save_discharge_model.compute - STEP_INPUT - Got input "discharge_opt_model_fp" of type "String". (Type check passed).
    [32m2021-03-19 01:24:43[0m - dagster - [34mDEBUG[0m - end_to_end_pipeline - 01f88437-2b8c-4581-9982-3f76523c2874 - 17436 - fit_and_save_discharge_model.compute - STEP_INPUT - Got input "model_params" of type "dict". (Type check passed).
    [32m2021-03-19 01:24:43[0m - dagster - [34mDEBUG[0m - end_to_end_pipeline - 01f88437-2b8c-4581-9982-3f76523c2874 - 17436 - fit_and_save_discharge_model.compute - STEP_OUTPUT - Yielded output "result" of type "Any". (Type check passed).
    [32m2021-03-19 01:24:43[0m - dagster - [34mDEBUG[0m - end_to_end_pipeline - 01f88437-2b8c-4581-9982-3f76523c2874 - 17436 - fit_and_save_discharge_model.compute - OBJECT_STORE_OPERATION - Stored intermediate object for output result in memory object store using pickle.
    [32m2021-03-19 01:24:43[0m - dagster - [34mDEBUG[0m - end_to_end_pipeline - 01f88437-2b8c-4581-9982-3f76523c2874 - 17436 - fit_and_save_discharge_model.compute - STEP_SUCCESS - Finished execution of step "fit_and_save_discharge_model.compute" in 5.44ms.
    [32m2021-03-19 01:24:43[0m - dagster - [34mDEBUG[0m - end_to_end_pipeline - 01f88437-2b8c-4581-9982-3f76523c2874 - 17436 - fit_and_save_pv_model.compute - STEP_START - Started execution of step "fit_and_save_pv_model.compute".
    [32m2021-03-19 01:24:43[0m - dagster - [34mDEBUG[0m - end_to_end_pipeline - 01f88437-2b8c-4581-9982-3f76523c2874 - 17436 - fit_and_save_pv_model.compute - OBJECT_STORE_OPERATION - Retrieved intermediate object for input intermediate_data_dir in memory object store using pickle.
    [32m2021-03-19 01:24:43[0m - dagster - [34mDEBUG[0m - end_to_end_pipeline - 01f88437-2b8c-4581-9982-3f76523c2874 - 17436 - fit_and_save_pv_model.compute - STEP_INPUT - Got input "intermediate_data_dir" of type "String". (Type check passed).
    [32m2021-03-19 01:24:43[0m - dagster - [34mDEBUG[0m - end_to_end_pipeline - 01f88437-2b8c-4581-9982-3f76523c2874 - 17436 - fit_and_save_pv_model.compute - STEP_INPUT - Got input "pv_model_fp" of type "String". (Type check passed).
    [32m2021-03-19 01:24:43[0m - dagster - [34mDEBUG[0m - end_to_end_pipeline - 01f88437-2b8c-4581-9982-3f76523c2874 - 17436 - fit_and_save_pv_model.compute - STEP_INPUT - Got input "model_params" of type "dict". (Type check passed).
    [32m2021-03-19 01:24:43[0m - dagster - [34mDEBUG[0m - end_to_end_pipeline - 01f88437-2b8c-4581-9982-3f76523c2874 - 17436 - fit_and_save_pv_model.compute - STEP_OUTPUT - Yielded output "result" of type "Any". (Type check passed).
    [32m2021-03-19 01:24:43[0m - dagster - [34mDEBUG[0m - end_to_end_pipeline - 01f88437-2b8c-4581-9982-3f76523c2874 - 17436 - fit_and_save_pv_model.compute - OBJECT_STORE_OPERATION - Stored intermediate object for output result in memory object store using pickle.
    [32m2021-03-19 01:24:43[0m - dagster - [34mDEBUG[0m - end_to_end_pipeline - 01f88437-2b8c-4581-9982-3f76523c2874 - 17436 - fit_and_save_pv_model.compute - STEP_SUCCESS - Finished execution of step "fit_and_save_pv_model.compute" in 6.29ms.
    [32m2021-03-19 01:24:43[0m - dagster - [34mDEBUG[0m - end_to_end_pipeline - 01f88437-2b8c-4581-9982-3f76523c2874 - 17436 - construct_battery_profile.compute - STEP_START - Started execution of step "construct_battery_profile.compute".
    [32m2021-03-19 01:24:43[0m - dagster - [34mDEBUG[0m - end_to_end_pipeline - 01f88437-2b8c-4581-9982-3f76523c2874 - 17436 - construct_battery_profile.compute - OBJECT_STORE_OPERATION - Retrieved intermediate object for input charge_model_success in memory object store using pickle.
    [32m2021-03-19 01:24:43[0m - dagster - [34mDEBUG[0m - end_to_end_pipeline - 01f88437-2b8c-4581-9982-3f76523c2874 - 17436 - construct_battery_profile.compute - OBJECT_STORE_OPERATION - Retrieved intermediate object for input discharge_model_success in memory object store using pickle.
    [32m2021-03-19 01:24:43[0m - dagster - [34mDEBUG[0m - end_to_end_pipeline - 01f88437-2b8c-4581-9982-3f76523c2874 - 17436 - construct_battery_profile.compute - OBJECT_STORE_OPERATION - Retrieved intermediate object for input intermediate_data_dir in memory object store using pickle.
    [32m2021-03-19 01:24:43[0m - dagster - [34mDEBUG[0m - end_to_end_pipeline - 01f88437-2b8c-4581-9982-3f76523c2874 - 17436 - construct_battery_profile.compute - STEP_INPUT - Got input "charge_model_success" of type "Bool". (Type check passed).
    [32m2021-03-19 01:24:43[0m - dagster - [34mDEBUG[0m - end_to_end_pipeline - 01f88437-2b8c-4581-9982-3f76523c2874 - 17436 - construct_battery_profile.compute - STEP_INPUT - Got input "discharge_model_success" of type "Bool". (Type check passed).
    [32m2021-03-19 01:24:43[0m - dagster - [34mDEBUG[0m - end_to_end_pipeline - 01f88437-2b8c-4581-9982-3f76523c2874 - 17436 - construct_battery_profile.compute - STEP_INPUT - Got input "intermediate_data_dir" of type "String". (Type check passed).
    [32m2021-03-19 01:24:43[0m - dagster - [34mDEBUG[0m - end_to_end_pipeline - 01f88437-2b8c-4581-9982-3f76523c2874 - 17436 - construct_battery_profile.compute - STEP_INPUT - Got input "raw_data_dir" of type "String". (Type check passed).
    [32m2021-03-19 01:24:43[0m - dagster - [34mDEBUG[0m - end_to_end_pipeline - 01f88437-2b8c-4581-9982-3f76523c2874 - 17436 - construct_battery_profile.compute - STEP_INPUT - Got input "discharge_opt_model_fp" of type "String". (Type check passed).
    [32m2021-03-19 01:24:43[0m - dagster - [34mDEBUG[0m - end_to_end_pipeline - 01f88437-2b8c-4581-9982-3f76523c2874 - 17436 - construct_battery_profile.compute - STEP_INPUT - Got input "pv_model_fp" of type "String". (Type check passed).
    [32m2021-03-19 01:24:43[0m - dagster - [34mDEBUG[0m - end_to_end_pipeline - 01f88437-2b8c-4581-9982-3f76523c2874 - 17436 - construct_battery_profile.compute - STEP_INPUT - Got input "start_time" of type "String". (Type check passed).
    [32m2021-03-19 01:24:46[0m - dagster - [34mDEBUG[0m - end_to_end_pipeline - 01f88437-2b8c-4581-9982-3f76523c2874 - 17436 - construct_battery_profile.compute - STEP_OUTPUT - Yielded output "result" of type "Any". (Type check passed).
    [32m2021-03-19 01:24:46[0m - dagster - [34mDEBUG[0m - end_to_end_pipeline - 01f88437-2b8c-4581-9982-3f76523c2874 - 17436 - construct_battery_profile.compute - OBJECT_STORE_OPERATION - Stored intermediate object for output result in memory object store using pickle.
    [32m2021-03-19 01:24:46[0m - dagster - [34mDEBUG[0m - end_to_end_pipeline - 01f88437-2b8c-4581-9982-3f76523c2874 - 17436 - construct_battery_profile.compute - STEP_SUCCESS - Finished execution of step "construct_battery_profile.compute" in 2.58s.
    [32m2021-03-19 01:24:46[0m - dagster - [34mDEBUG[0m - end_to_end_pipeline - 01f88437-2b8c-4581-9982-3f76523c2874 - 17436 - check_and_save_battery_profile.compute - STEP_START - Started execution of step "check_and_save_battery_profile.compute".
    [32m2021-03-19 01:24:46[0m - dagster - [34mDEBUG[0m - end_to_end_pipeline - 01f88437-2b8c-4581-9982-3f76523c2874 - 17436 - check_and_save_battery_profile.compute - OBJECT_STORE_OPERATION - Retrieved intermediate object for input s_battery_profile in memory object store using pickle.
    [32m2021-03-19 01:24:46[0m - dagster - [34mDEBUG[0m - end_to_end_pipeline - 01f88437-2b8c-4581-9982-3f76523c2874 - 17436 - check_and_save_battery_profile.compute - STEP_INPUT - Got input "s_battery_profile" of type "Any". (Type check passed).
    [32m2021-03-19 01:24:46[0m - dagster - [34mDEBUG[0m - end_to_end_pipeline - 01f88437-2b8c-4581-9982-3f76523c2874 - 17436 - check_and_save_battery_profile.compute - STEP_INPUT - Got input "output_data_dir" of type "String". (Type check passed).
    [32m2021-03-19 01:24:46[0m - dagster - [34mDEBUG[0m - end_to_end_pipeline - 01f88437-2b8c-4581-9982-3f76523c2874 - 17436 - check_and_save_battery_profile.compute - STEP_OUTPUT - Yielded output "result" of type "Any". (Type check passed).
    [32m2021-03-19 01:24:46[0m - dagster - [34mDEBUG[0m - end_to_end_pipeline - 01f88437-2b8c-4581-9982-3f76523c2874 - 17436 - check_and_save_battery_profile.compute - OBJECT_STORE_OPERATION - Stored intermediate object for output result in memory object store using pickle.
    [32m2021-03-19 01:24:46[0m - dagster - [34mDEBUG[0m - end_to_end_pipeline - 01f88437-2b8c-4581-9982-3f76523c2874 - 17436 - check_and_save_battery_profile.compute - STEP_SUCCESS - Finished execution of step "check_and_save_battery_profile.compute" in 15ms.
    [32m2021-03-19 01:24:46[0m - dagster - [34mDEBUG[0m - end_to_end_pipeline - 01f88437-2b8c-4581-9982-3f76523c2874 - 17436 - ENGINE_EVENT - Finished steps in process (pid: 17436) in 2m4s
    [32m2021-03-19 01:24:46[0m - dagster - [34mDEBUG[0m - end_to_end_pipeline - 01f88437-2b8c-4581-9982-3f76523c2874 - 17436 - PIPELINE_SUCCESS - Finished execution of pipeline "end_to_end_pipeline".
    




    <dagster.core.execution.results.PipelineExecutionResult at 0x1d1fa542ac0>



<br>

We'll then visualise the latest charging profile

```python
df_latest_submission = pd.read_csv('../data/output/latest_submission.csv')

s_latest_submission = df_latest_submission.set_index('datetime')['charge_MW']
s_latest_submission.index = pd.to_datetime(s_latest_submission.index)

# Plotting
fig, ax = plt.subplots(dpi=250)

s_latest_submission.plot(ax=ax)

ax.set_xlabel('')
ax.set_ylabel('Charge (MW)')
hlp.hide_spines(ax)
fig.tight_layout()
fig.savefig('../img/latest_submission.png', dpi=250)
```


![png](img/nbs/output_9_0.png)


<br>

Finally we'll export the relevant code to our `batopt` module
