# Pipeline



```python
#exports
import numpy as np
import pandas as pd

from dagster import execute_pipeline, pipeline, solid, Field

from batopt import clean, discharge
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
def clean_data(_, loaded_data, intermediate_data_dir: str):
    # Cleaning
    cleaned_data = dict()

    cleaned_data['pv'] = (loaded_data['pv']
                          .pipe(clean.interpolate_missing_panel_temps, loaded_data['weather'])
                          .pipe(clean.interpolate_missing_site_irradiance, loaded_data['weather'])
                          .pipe(clean.interpolate_missing_site_power)
                         )
    cleaned_data['weather'] = clean.interpolate_missing_weather_solar(loaded_data['pv'], loaded_data['weather'])
    cleaned_data['demand'] = loaded_data['demand']
    
    # Saving
    cleaned_data['pv'].to_csv(f'{intermediate_data_dir}/pv_cleaned.csv')
    cleaned_data['demand'].to_csv(f'{intermediate_data_dir}/demand_cleaned.csv')
    cleaned_data['weather'].to_csv(f'{intermediate_data_dir}/weather_cleaned.csv')
            
    return intermediate_data_dir

@solid()
def fit_and_save_discharge_model(_, intermediate_data_dir: str, discharge_opt_model_fp: str, model_params: dict):
    X, y = discharge.prepare_training_input_data(intermediate_data_dir)
    discharge.fit_and_save_model(X, y, discharge_opt_model_fp, **model_params)
    
    return 

@solid()
def construct_battery_profile(_, cleaned_data_dir: str, raw_data_dir: str, discharge_opt_model_fp: str):
    s_discharge_profile = discharge.optimise_latest_test_discharge_profile(raw_data_dir, cleaned_data_dir, discharge_opt_model_fp)
    
    s_battery_profile = s_discharge_profile
    
    return s_battery_profile
```

<br>

Then we'll combine them in a pipeline

```python
@pipeline
def end_to_end_pipeline(): 
    loaded_data = load_data()
    cleaned_data_dir = clean_data(loaded_data)
    
    fit_and_save_discharge_model(cleaned_data_dir)
    s_battery_profile = construct_battery_profile(cleaned_data_dir)
    # Should use `great expectations` to check that the battery profile doesnt break the constraints
```

<br>

Which we'll now run a test with

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
                'intermediate_data_dir': '../data/intermediate',
            },
        },
        'fit_and_save_discharge_model': {
            'inputs': {
                'discharge_opt_model_fp': '../models/discharge_opt.sav',
                'model_params': {
                    'criterion': 'mse',
                    'max_depth': 10,
                    'min_samples_leaf': 4,
                    'min_samples_split': 2,
                    'n_estimators': 100                    
                }
            },
        },
        'construct_battery_profile': {
            'inputs': {
                'raw_data_dir': '../data/raw',
                'discharge_opt_model_fp': '../models/discharge_opt.sav',
            },
        },
    }
}

execute_pipeline(end_to_end_pipeline, run_config=run_config)
```

    [32m2021-02-05 16:31:18[0m - dagster - [34mDEBUG[0m - end_to_end_pipeline - 84fbd0bf-b030-4059-bd61-1023b3c6c933 - 22040 - ENGINE_EVENT - Starting initialization of resources [asset_store].
    [32m2021-02-05 16:31:18[0m - dagster - [34mDEBUG[0m - end_to_end_pipeline - 84fbd0bf-b030-4059-bd61-1023b3c6c933 - 22040 - ENGINE_EVENT - Finished initialization of resources [asset_store].
    [32m2021-02-05 16:31:18[0m - dagster - [34mDEBUG[0m - end_to_end_pipeline - 84fbd0bf-b030-4059-bd61-1023b3c6c933 - 22040 - PIPELINE_START - Started execution of pipeline "end_to_end_pipeline".
    [32m2021-02-05 16:31:18[0m - dagster - [34mDEBUG[0m - end_to_end_pipeline - 84fbd0bf-b030-4059-bd61-1023b3c6c933 - 22040 - ENGINE_EVENT - Executing steps in process (pid: 22040)
    [32m2021-02-05 16:31:18[0m - dagster - [34mDEBUG[0m - end_to_end_pipeline - 84fbd0bf-b030-4059-bd61-1023b3c6c933 - 22040 - load_data.compute - STEP_START - Started execution of step "load_data.compute".
    [32m2021-02-05 16:31:18[0m - dagster - [34mDEBUG[0m - end_to_end_pipeline - 84fbd0bf-b030-4059-bd61-1023b3c6c933 - 22040 - load_data.compute - STEP_INPUT - Got input "raw_data_dir" of type "String". (Type check passed).
    [32m2021-02-05 16:31:18[0m - dagster - [34mDEBUG[0m - end_to_end_pipeline - 84fbd0bf-b030-4059-bd61-1023b3c6c933 - 22040 - load_data.compute - STEP_OUTPUT - Yielded output "result" of type "Any". (Type check passed).
    [32m2021-02-05 16:31:18[0m - dagster - [34mDEBUG[0m - end_to_end_pipeline - 84fbd0bf-b030-4059-bd61-1023b3c6c933 - 22040 - load_data.compute - OBJECT_STORE_OPERATION - Stored intermediate object for output result in memory object store using pickle.
    [32m2021-02-05 16:31:18[0m - dagster - [34mDEBUG[0m - end_to_end_pipeline - 84fbd0bf-b030-4059-bd61-1023b3c6c933 - 22040 - load_data.compute - STEP_SUCCESS - Finished execution of step "load_data.compute" in 124ms.
    [32m2021-02-05 16:31:18[0m - dagster - [34mDEBUG[0m - end_to_end_pipeline - 84fbd0bf-b030-4059-bd61-1023b3c6c933 - 22040 - clean_data.compute - STEP_START - Started execution of step "clean_data.compute".
    [32m2021-02-05 16:31:18[0m - dagster - [34mDEBUG[0m - end_to_end_pipeline - 84fbd0bf-b030-4059-bd61-1023b3c6c933 - 22040 - clean_data.compute - OBJECT_STORE_OPERATION - Retrieved intermediate object for input loaded_data in memory object store using pickle.
    [32m2021-02-05 16:31:18[0m - dagster - [34mDEBUG[0m - end_to_end_pipeline - 84fbd0bf-b030-4059-bd61-1023b3c6c933 - 22040 - clean_data.compute - STEP_INPUT - Got input "loaded_data" of type "Any". (Type check passed).
    [32m2021-02-05 16:31:18[0m - dagster - [34mDEBUG[0m - end_to_end_pipeline - 84fbd0bf-b030-4059-bd61-1023b3c6c933 - 22040 - clean_data.compute - STEP_INPUT - Got input "intermediate_data_dir" of type "String". (Type check passed).
    [32m2021-02-05 16:31:47[0m - dagster - [34mDEBUG[0m - end_to_end_pipeline - 84fbd0bf-b030-4059-bd61-1023b3c6c933 - 22040 - clean_data.compute - STEP_OUTPUT - Yielded output "result" of type "Any". (Type check passed).
    [32m2021-02-05 16:31:47[0m - dagster - [34mDEBUG[0m - end_to_end_pipeline - 84fbd0bf-b030-4059-bd61-1023b3c6c933 - 22040 - clean_data.compute - OBJECT_STORE_OPERATION - Stored intermediate object for output result in memory object store using pickle.
    [32m2021-02-05 16:31:47[0m - dagster - [34mDEBUG[0m - end_to_end_pipeline - 84fbd0bf-b030-4059-bd61-1023b3c6c933 - 22040 - clean_data.compute - STEP_SUCCESS - Finished execution of step "clean_data.compute" in 29.07s.
    [32m2021-02-05 16:31:47[0m - dagster - [34mDEBUG[0m - end_to_end_pipeline - 84fbd0bf-b030-4059-bd61-1023b3c6c933 - 22040 - construct_battery_profile.compute - STEP_START - Started execution of step "construct_battery_profile.compute".
    [32m2021-02-05 16:31:47[0m - dagster - [34mDEBUG[0m - end_to_end_pipeline - 84fbd0bf-b030-4059-bd61-1023b3c6c933 - 22040 - construct_battery_profile.compute - OBJECT_STORE_OPERATION - Retrieved intermediate object for input cleaned_data_dir in memory object store using pickle.
    [32m2021-02-05 16:31:47[0m - dagster - [34mDEBUG[0m - end_to_end_pipeline - 84fbd0bf-b030-4059-bd61-1023b3c6c933 - 22040 - construct_battery_profile.compute - STEP_INPUT - Got input "cleaned_data_dir" of type "String". (Type check passed).
    [32m2021-02-05 16:31:47[0m - dagster - [34mDEBUG[0m - end_to_end_pipeline - 84fbd0bf-b030-4059-bd61-1023b3c6c933 - 22040 - construct_battery_profile.compute - STEP_INPUT - Got input "raw_data_dir" of type "String". (Type check passed).
    [32m2021-02-05 16:31:47[0m - dagster - [34mDEBUG[0m - end_to_end_pipeline - 84fbd0bf-b030-4059-bd61-1023b3c6c933 - 22040 - construct_battery_profile.compute - STEP_INPUT - Got input "discharge_opt_model_fp" of type "String". (Type check passed).
    [32m2021-02-05 16:31:48[0m - dagster - [34mDEBUG[0m - end_to_end_pipeline - 84fbd0bf-b030-4059-bd61-1023b3c6c933 - 22040 - construct_battery_profile.compute - STEP_OUTPUT - Yielded output "result" of type "Any". (Type check passed).
    [32m2021-02-05 16:31:48[0m - dagster - [34mDEBUG[0m - end_to_end_pipeline - 84fbd0bf-b030-4059-bd61-1023b3c6c933 - 22040 - construct_battery_profile.compute - OBJECT_STORE_OPERATION - Stored intermediate object for output result in memory object store using pickle.
    [32m2021-02-05 16:31:48[0m - dagster - [34mDEBUG[0m - end_to_end_pipeline - 84fbd0bf-b030-4059-bd61-1023b3c6c933 - 22040 - construct_battery_profile.compute - STEP_SUCCESS - Finished execution of step "construct_battery_profile.compute" in 660ms.
    [32m2021-02-05 16:31:48[0m - dagster - [34mDEBUG[0m - end_to_end_pipeline - 84fbd0bf-b030-4059-bd61-1023b3c6c933 - 22040 - fit_and_save_discharge_model.compute - STEP_START - Started execution of step "fit_and_save_discharge_model.compute".
    [32m2021-02-05 16:31:48[0m - dagster - [34mDEBUG[0m - end_to_end_pipeline - 84fbd0bf-b030-4059-bd61-1023b3c6c933 - 22040 - fit_and_save_discharge_model.compute - OBJECT_STORE_OPERATION - Retrieved intermediate object for input intermediate_data_dir in memory object store using pickle.
    [32m2021-02-05 16:31:48[0m - dagster - [34mDEBUG[0m - end_to_end_pipeline - 84fbd0bf-b030-4059-bd61-1023b3c6c933 - 22040 - fit_and_save_discharge_model.compute - STEP_INPUT - Got input "intermediate_data_dir" of type "String". (Type check passed).
    [32m2021-02-05 16:31:48[0m - dagster - [34mDEBUG[0m - end_to_end_pipeline - 84fbd0bf-b030-4059-bd61-1023b3c6c933 - 22040 - fit_and_save_discharge_model.compute - STEP_INPUT - Got input "discharge_opt_model_fp" of type "String". (Type check passed).
    [32m2021-02-05 16:31:48[0m - dagster - [34mDEBUG[0m - end_to_end_pipeline - 84fbd0bf-b030-4059-bd61-1023b3c6c933 - 22040 - fit_and_save_discharge_model.compute - STEP_INPUT - Got input "model_params" of type "dict". (Type check passed).
    [32m2021-02-05 16:31:51[0m - dagster - [34mDEBUG[0m - end_to_end_pipeline - 84fbd0bf-b030-4059-bd61-1023b3c6c933 - 22040 - fit_and_save_discharge_model.compute - STEP_OUTPUT - Yielded output "result" of type "Any". (Type check passed).
    [32m2021-02-05 16:31:51[0m - dagster - [34mDEBUG[0m - end_to_end_pipeline - 84fbd0bf-b030-4059-bd61-1023b3c6c933 - 22040 - fit_and_save_discharge_model.compute - OBJECT_STORE_OPERATION - Stored intermediate object for output result in memory object store using pickle.
    [32m2021-02-05 16:31:51[0m - dagster - [34mDEBUG[0m - end_to_end_pipeline - 84fbd0bf-b030-4059-bd61-1023b3c6c933 - 22040 - fit_and_save_discharge_model.compute - STEP_SUCCESS - Finished execution of step "fit_and_save_discharge_model.compute" in 2.72s.
    [32m2021-02-05 16:31:51[0m - dagster - [34mDEBUG[0m - end_to_end_pipeline - 84fbd0bf-b030-4059-bd61-1023b3c6c933 - 22040 - ENGINE_EVENT - Finished steps in process (pid: 22040) in 32.61s
    [32m2021-02-05 16:31:51[0m - dagster - [34mDEBUG[0m - end_to_end_pipeline - 84fbd0bf-b030-4059-bd61-1023b3c6c933 - 22040 - PIPELINE_SUCCESS - Finished execution of pipeline "end_to_end_pipeline".
    




    <dagster.core.execution.results.PipelineExecutionResult at 0x1abe5f7cd90>



<br>

Finally we'll export the relevant code to our `batopt` module
