# Pipeline



```python
#exports
import numpy as np
import pandas as pd

from dagster import execute_pipeline, pipeline, solid, Field

from batopt import clean
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
            
    return cleaned_data
```

<br>

Then we'll combine them in a pipeline

```python
@pipeline
def end_to_end_pipeline(): 
    loaded_data = load_data()
    cleaned_data = clean_data(loaded_data)
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
    }
}

execute_pipeline(end_to_end_pipeline, run_config=run_config)
```

    [32m2021-01-29 10:43:45[0m - dagster - [34mDEBUG[0m - end_to_end_pipeline - bdac1336-c21b-46a1-af7e-f40d144fe459 - 22680 - ENGINE_EVENT - Starting initialization of resources [asset_store].
    [32m2021-01-29 10:43:45[0m - dagster - [34mDEBUG[0m - end_to_end_pipeline - bdac1336-c21b-46a1-af7e-f40d144fe459 - 22680 - ENGINE_EVENT - Finished initialization of resources [asset_store].
    [32m2021-01-29 10:43:45[0m - dagster - [34mDEBUG[0m - end_to_end_pipeline - bdac1336-c21b-46a1-af7e-f40d144fe459 - 22680 - PIPELINE_START - Started execution of pipeline "end_to_end_pipeline".
    [32m2021-01-29 10:43:45[0m - dagster - [34mDEBUG[0m - end_to_end_pipeline - bdac1336-c21b-46a1-af7e-f40d144fe459 - 22680 - ENGINE_EVENT - Executing steps in process (pid: 22680)
    [32m2021-01-29 10:43:45[0m - dagster - [34mDEBUG[0m - end_to_end_pipeline - bdac1336-c21b-46a1-af7e-f40d144fe459 - 22680 - load_data.compute - STEP_START - Started execution of step "load_data.compute".
    [32m2021-01-29 10:43:45[0m - dagster - [34mDEBUG[0m - end_to_end_pipeline - bdac1336-c21b-46a1-af7e-f40d144fe459 - 22680 - load_data.compute - STEP_INPUT - Got input "raw_data_dir" of type "String". (Type check passed).
    [32m2021-01-29 10:43:45[0m - dagster - [34mDEBUG[0m - end_to_end_pipeline - bdac1336-c21b-46a1-af7e-f40d144fe459 - 22680 - load_data.compute - STEP_OUTPUT - Yielded output "result" of type "Any". (Type check passed).
    [32m2021-01-29 10:43:45[0m - dagster - [34mDEBUG[0m - end_to_end_pipeline - bdac1336-c21b-46a1-af7e-f40d144fe459 - 22680 - load_data.compute - OBJECT_STORE_OPERATION - Stored intermediate object for output result in memory object store using pickle.
    [32m2021-01-29 10:43:45[0m - dagster - [34mDEBUG[0m - end_to_end_pipeline - bdac1336-c21b-46a1-af7e-f40d144fe459 - 22680 - load_data.compute - STEP_SUCCESS - Finished execution of step "load_data.compute" in 155ms.
    [32m2021-01-29 10:43:45[0m - dagster - [34mDEBUG[0m - end_to_end_pipeline - bdac1336-c21b-46a1-af7e-f40d144fe459 - 22680 - clean_data.compute - STEP_START - Started execution of step "clean_data.compute".
    [32m2021-01-29 10:43:45[0m - dagster - [34mDEBUG[0m - end_to_end_pipeline - bdac1336-c21b-46a1-af7e-f40d144fe459 - 22680 - clean_data.compute - OBJECT_STORE_OPERATION - Retrieved intermediate object for input loaded_data in memory object store using pickle.
    [32m2021-01-29 10:43:45[0m - dagster - [34mDEBUG[0m - end_to_end_pipeline - bdac1336-c21b-46a1-af7e-f40d144fe459 - 22680 - clean_data.compute - STEP_INPUT - Got input "loaded_data" of type "Any". (Type check passed).
    [32m2021-01-29 10:43:45[0m - dagster - [34mDEBUG[0m - end_to_end_pipeline - bdac1336-c21b-46a1-af7e-f40d144fe459 - 22680 - clean_data.compute - STEP_INPUT - Got input "intermediate_data_dir" of type "String". (Type check passed).
    [32m2021-01-29 10:44:24[0m - dagster - [34mDEBUG[0m - end_to_end_pipeline - bdac1336-c21b-46a1-af7e-f40d144fe459 - 22680 - clean_data.compute - STEP_OUTPUT - Yielded output "result" of type "Any". (Type check passed).
    [32m2021-01-29 10:44:24[0m - dagster - [34mDEBUG[0m - end_to_end_pipeline - bdac1336-c21b-46a1-af7e-f40d144fe459 - 22680 - clean_data.compute - OBJECT_STORE_OPERATION - Stored intermediate object for output result in memory object store using pickle.
    [32m2021-01-29 10:44:24[0m - dagster - [34mDEBUG[0m - end_to_end_pipeline - bdac1336-c21b-46a1-af7e-f40d144fe459 - 22680 - clean_data.compute - STEP_SUCCESS - Finished execution of step "clean_data.compute" in 38.81s.
    [32m2021-01-29 10:44:24[0m - dagster - [34mDEBUG[0m - end_to_end_pipeline - bdac1336-c21b-46a1-af7e-f40d144fe459 - 22680 - ENGINE_EVENT - Finished steps in process (pid: 22680) in 38.99s
    [32m2021-01-29 10:44:24[0m - dagster - [34mDEBUG[0m - end_to_end_pipeline - bdac1336-c21b-46a1-af7e-f40d144fe459 - 22680 - PIPELINE_SUCCESS - Finished execution of pipeline "end_to_end_pipeline".
    




    <dagster.core.execution.results.PipelineExecutionResult at 0x17f0e037c40>



<br>

Finally we'll export the relevant code to our `batopt` module
