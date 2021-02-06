# PV Forecasting



### Imports

```python
#exports
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns

from moepy.lowess import quantile_model
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

from batopt import clean
from batopt.discharge import sample_random_day

import FEAutils as hlp
```

```python
# Should do some investigation of how the panel temp influences performance
```

```python
raw_data_dir = '../data/raw'
intermediate_data_dir = '../data/intermediate'
cache_data_dir = '../data/nb-cache'
```

```python
df = clean.combine_training_datasets(intermediate_data_dir).interpolate(limit=1)

df.head()
```




| ('Unnamed: 0_level_0', 'datetime')   |   ('demand_MW', 'Unnamed: 1_level_1') |   ('irradiance_Wm-2', 'Unnamed: 2_level_1') |   ('panel_temp_C', 'Unnamed: 3_level_1') |   ('pv_power_mw', 'Unnamed: 4_level_1') |   ('solar_location1', 'Unnamed: 5_level_1') |   ('solar_location2', 'Unnamed: 6_level_1') |   ('solar_location3', 'Unnamed: 7_level_1') |   ('solar_location4', 'Unnamed: 8_level_1') |   ('solar_location5', 'Unnamed: 9_level_1') |   ('solar_location6', 'Unnamed: 10_level_1') |   ('temp_location1', 'Unnamed: 11_level_1') |   ('temp_location2', 'Unnamed: 12_level_1') |   ('temp_location3', 'Unnamed: 13_level_1') |   ('temp_location4', 'Unnamed: 14_level_1') |   ('temp_location5', 'Unnamed: 15_level_1') |   ('temp_location6', 'Unnamed: 16_level_1') |
|:-------------------------------------|--------------------------------------:|--------------------------------------------:|-----------------------------------------:|----------------------------------------:|--------------------------------------------:|--------------------------------------------:|--------------------------------------------:|--------------------------------------------:|--------------------------------------------:|---------------------------------------------:|--------------------------------------------:|--------------------------------------------:|--------------------------------------------:|--------------------------------------------:|--------------------------------------------:|--------------------------------------------:|
| 2017-11-03 00:00:00+00:00            |                                  2.19 |                                           0 |                                     7.05 |                                       0 |                                           0 |                                           0 |                                           0 |                                           0 |                                           0 |                                            0 |                                       8.56  |                                       9.64  |                                        7.46 |                                       6.68  |                                      13.09  |                                       13.2  |
| 2017-11-03 00:30:00+00:00            |                                  2.14 |                                           0 |                                     7.38 |                                       0 |                                           0 |                                           0 |                                           0 |                                           0 |                                           0 |                                            0 |                                       8.625 |                                       9.675 |                                        7.3  |                                       6.475 |                                      13.15  |                                       13.26 |
| 2017-11-03 01:00:00+00:00            |                                  2.01 |                                           0 |                                     7.7  |                                       0 |                                           0 |                                           0 |                                           0 |                                           0 |                                           0 |                                            0 |                                       8.69  |                                       9.71  |                                        7.14 |                                       6.27  |                                      13.21  |                                       13.32 |
| 2017-11-03 01:30:00+00:00            |                                  1.87 |                                           0 |                                     7.48 |                                       0 |                                           0 |                                           0 |                                           0 |                                           0 |                                           0 |                                            0 |                                       8.715 |                                       9.72  |                                        7    |                                       6.09  |                                      13.255 |                                       13.34 |
| 2017-11-03 02:00:00+00:00            |                                  1.86 |                                           0 |                                     7.2  |                                       0 |                                           0 |                                           0 |                                           0 |                                           0 |                                           0 |                                            0 |                                       8.74  |                                       9.73  |                                        6.86 |                                       5.91  |                                      13.3   |                                       13.36 |</div>



Correlations between the solar variables:

```python
solar_cols = [c for c in df.columns if 'solar_location' in c]
solar_cols.append('irradiance_Wm-2')
solar_cols.append('panel_temp_C')
solar_cols.append('pv_power_mw')

fig, ax = plt.subplots(dpi=250)
df_solar = df.filter(solar_cols).copy()
ax = sns.heatmap(df_solar.corr(), cmap='viridis')
fig.savefig('../img/solar_corrplot.png')
```


![png](img/nbs/output_6_0.png)


As in the demand data, estimating the quantiles for the solar PV output:

```python
#exports
def estimate_daily_solar_quantiles(x, y, x_pred = np.linspace(0, 23.5, 100), **model_kwargs):
    # Fitting the model
    df_quantiles = quantile_model(x, y, x_pred=x_pred, **model_kwargs)

    # Cleaning names and sorting for plotting
    df_quantiles.columns = [f'p{int(col*100)}' for col in df_quantiles.columns]
    df_quantiles = df_quantiles[df_quantiles.columns[::-1]]
    
    return df_quantiles

dts = df.index.tz_convert('Europe/London')
x = np.array(dts.hour + dts.minute/60)
y = df['pv_power_mw'].values

rerun_daily_solar_model = False
daily_solar_filename = 'daily_solar_quantile_model_results.csv'

if (rerun_daily_solar_model == True) or (daily_solar_filename not in os.listdir(cache_data_dir)):
    df_quantiles = estimate_daily_solar_quantiles(x, y, frac=0.2, num_fits=48, robust_iters=3)
    df_quantiles.to_csv(f'{cache_data_dir}/{daily_solar_filename}')
else:
    df_quantiles = pd.read_csv(f'{cache_data_dir}/{daily_solar_filename}', index_col='x')
```

And plotting

```python
x_jittered = x + (np.random.uniform(size=len(x)) - 0.5)/2.5

# Plotting
fig, ax = plt.subplots(dpi=250)

ax.scatter(x_jittered, y, s=0.2, color='k', alpha=0.5)
df_quantiles.plot(cmap='viridis', legend=False, ax=ax)

hlp.hide_spines(ax)
ax.legend(frameon=False, bbox_to_anchor=(1, 0.9), title='Percentiles')
ax.set_xlabel('Time of Day')
ax.set_ylabel('Demand (MW)')
ax.set_xlim(0, 24)
ax.set_ylim(0, 4)

fig.savefig('../img/daily_solar_profile.png')
```


![png](img/nbs/output_10_0.png)


## Proportion of days during which we can fully charge the battery

It may be useful to know the proportion of days during which the battery can be fully charged. 

```python
df_solar_hrs = df.between_time('00:00:00', '15:00:00')
pv_generation = df_solar_hrs.groupby(df_solar_hrs.index.date).sum()['pv_power_mw']*0.5 # available daily energy from PV

fig, ax = plt.subplots()
ax.hist(pv_generation, bins=20)
plt.show()

prop = np.sum(pv_generation >= 6)/pv_generation.size
print("Proportion of days where solar generation exceeds 6 MWh: {:.2f}%".format(prop*100))
```


![png](img/nbs/output_12_0.png)


    Proportion of days where solar generation exceeds 6 MWh: 69.08%
    

## Optimal charging with perfect foresight

We will now develop an algorithm to determine the optimal charging schedule given a perfect solar forecast. 

The scoring function for the generation component rewards us taking as much energy as possible from solar PV. The proportion of energy from PV for a day $d$ is given by $$p_{d,1} = \frac{\sum{P_{d,k}}}{\sum{B_{d,k}}}$$ where we are summing over all periods $k$. An equivalent equation is applies for $p_{d,2}$ which is the energy that is drawn from the grid. The scoring function rewards $p_{d,1}$ over $p_{d,2}$ in a ratio of 3 to 1. 

Any schedule which fully exploits the solar PV potential until the battery is charged is equally good in terms of the scoring function. However, it may be worth considering methods which give a smoother charge profile for the purposes of producing a robust model for unseen days.

In addition, we need to have a method of intelligently allocating charge when the solar PV potential is less than the capacity of the battery.

Some possible methods for this:

- Naively reallocate over the middle of they day (say 09:00--15:00)
- Add charge to periods where charge has already been committed.
- Use a forecast for PV output and allocate charge proportionally to the forecast.

```python
#exports
def extract_solar_profile(s_solar_sample_dt, start_time='00:00', end_time='15:00'):
    dt = str(s_solar_sample_dt.index[0].date())
    solar_profile = s_solar_sample_dt[f'{dt} {start_time}':f'{dt} {end_time}'].values

    return solar_profile
```

```python
s_pv = df['pv_power_mw']
solar_profile = sample_random_day(s_pv).pipe(extract_solar_profile)

plt.plot(solar_profile)
```




    [<matplotlib.lines.Line2D at 0x15b019730>]




![png](img/nbs/output_15_1.png)


For perfect foresight, any schedule that draws all of the available solar power or 6 MWh (if the total solar production exceeds 6 MWh) is equally good. 

This first approach will aim to draw greedily from  until 6 MWh is satisfied, or all of the solar production has been expended.

In cases where there is not enough solar PV to fill the battery, we will then uniformly add the remaining capacity across all periods.

**Note: this seems to work on this dataset but won't if there is a very large spike in solar PV, such topping up uniformly causes a constraint to be violated. It also may not work if the number of periods over which we top up is decreased.**

```python
#exports

def charge_profile_greedy(solar_profile, capacity=6, initial_charge=0, max_charge_rate=2.5, time_unit=0.5):
    order = np.flip(np.argsort(random_day))
    charge = initial_charge
    solution = np.zeros(len(solar_profile))
    for i in order:
        solar_available = np.minimum(solar_profile[i], max_charge_rate)
        solar_available = min(solar_available, (capacity - charge)/time_unit) 
        solution[i] = solar_available
        charge = np.sum(solution)*time_unit
        if charge > capacity:
            break
    return solution

def topup_charge_naive(charge_profile, capacity=6, time_unit=0.5, period_start=16, period_end=30):
    charge = np.sum(charge_profile)*time_unit
    spare_cap = capacity - charge
    topup_value = spare_cap/((period_end-period_start)*time_unit)
    new_profile = np.copy(charge_profile)
    new_profile[period_start:period_end] += topup_value # Add topup_value uniformly between start and end periods
    return new_profile

def scale_charge(charge_profile, capacity=6, time_unit=0.5):
    """
    Scale a charging profile to sum to capacity/time_unit while maintaining its shape
    """
    charge_profile = (capacity/time_unit)*charge_profile/np.sum(charge_profile)
    return charge_profile

def optimal_charge_profile(solar_profile, capacity=6, time_unit=0.5, max_charge_rate=2.5):
    solution = charge_profile_greedy(solar_profile)
    solution = topup_charge_naive(solution)
    assert np.isclose(np.sum(solution), capacity/time_unit), "Does not meet capacity constraint".format(np.sum(solution)) 
    assert np.all(solution <= max_charge_rate), "Does not meet max charge rate constraint. Max is {}".format(np.max(solution))
    return solution

random_day = sample_random_day(s_pv).pipe(extract_solar_profile)
x = optimal_charge_profile(random_day) # Note there is sometimes a rounding error here
```

The danger with this method is that it can be quite spiky. I wonder if this (a) makes the function difficult to learn (b) is too risky as compared with hedging bets with a more smoother approach. 

```python
# TODO: consider a different optimal charging algorithm that gives a smoother charge profile
# ^^^^  perhaps similar to the load flattening idea
```

Below we will construct the charge profiles for the dataset, which we will then use as target values. 

```python
#exports
def construct_charge_s(s_pv, start_time='00:00', end_time='15:00'):
    s_charge = pd.Series(index=s_pv.index, dtype=float).fillna(0)

    for dt in s_pv.index.strftime('%Y-%m-%d').unique():
        solar_profile = s_pv[dt].pipe(extract_solar_profile)
        charge_profile = optimal_charge_profile(solar_profile)
        s_charge[f'{dt} {start_time}':f'{dt} {end_time}'] = charge_profile

    return s_charge

s_charge = construct_charge_s(s_pv)
```

```python
s_charge.iloc[:48*7].plot()
```




    <AxesSubplot:xlabel='datetime'>




![png](img/nbs/output_22_1.png)


With the greedy algorithm we can analyse the periods during which charging occurs:

```python
s_charge.groupby(s_charge.index.time).sum().plot()
```




    <AxesSubplot:xlabel='time'>




![png](img/nbs/output_24_1.png)


Unsurprisingly we never charge before 5am. We can therefore truncate our training to just look at 05:00--15:30. 

This algorithm does not guarantee that we will fill the battery: on days when less than 6 MWh of solar PV are available, we still need to charge the battery to full. With perfect foresight it would not matter how we schedule the non-PV charging. However, for this task we really want to pick periods with the highest probability of solar PV generation (e.g. not the middle of the night). 

```python
# TODO: optimise with respect to a PV forecast.
# ^^^^  this would give a better method for scheduling when the expected solar PV is low.
```

### Model development: charging

Following the same structure as battery discharge, we will aim to predict the optimal charge schedule. 

```python
#exports 
def construct_df_charge_features(df):
    # Filtering for the temperature weather data
    df_features = df[df.columns[df.columns.str.contains('temp_location')]].copy()
    
    # Adding lagged demand
    df_features['demand_7d_lag'] = df['demand_MW'].shift(48*7)

    # Adding datetime features
    dts = df_features.index.tz_convert('Europe/London') # We want to use the 'behavioural' timezone

    df_features['weekend'] = dts.dayofweek.isin([5, 6]).astype(int)
    df_features['hour'] = dts.hour + dts.minute/60
    df_features['doy'] = dts.dayofyear
    df_features['dow'] = dts.dayofweek
    
    # Removing NaN values
    df_features = df_features.dropna()
    
    return df_features

#exports
def extract_charging_datetimes(df, start_hour=5, end_hour=15):
    hour = df.index.hour + df.index.minute/60
    charging_datetimes = df.index[(hour>=start_hour) & (hour<=end_hour)]
    
    return charging_datetimes


```

```python
df_features = construct_df_charge_features(df)
charging_datetimes = extract_charging_datetimes(df_features)

X = df_features.loc[charging_datetimes].values
y = s_charge.loc[charging_datetimes].values
```

```python
df_pred = clean.generate_kfold_preds(X, y, LinearRegression(), index=charging_datetimes)

df_pred.head()
```




| ('Unnamed: 0_level_0', 'datetime')   |   ('pred', 'Unnamed: 1_level_1') |   ('true', 'Unnamed: 2_level_1') |
|:-------------------------------------|---------------------------------:|---------------------------------:|
| 2017-11-10 05:00:00+00:00            |                         0.038502 |                                0 |
| 2017-11-10 05:30:00+00:00            |                         0.119941 |                                0 |
| 2017-11-10 06:00:00+00:00            |                         0.153795 |                                0 |
| 2017-11-10 06:30:00+00:00            |                         0.238854 |                                0 |
| 2017-11-10 07:00:00+00:00            |                         0.35312  |                                0 |</div>



We need to fix the predictions such that they satisfy the battery constraints. We will do this in the same way as applied in the battery discharge component, first clipping the charge rate to be between 0--2.5MW, then normalising such that the total charge sums to 6 MWh.

```python
#exports
def normalise_total_charge(s_pred, charge=6., time_unit=0.5):
    s_daily_charge = s_pred.groupby(s_pred.index.date).sum()

    for date, total_charge in s_daily_charge.items():
        s_pred.loc[str(date)] *= (charge/(time_unit*total_charge))
        
    return s_pred    

clip_charge_rate = lambda s_pred, max_rate=2.5, min_rate=0: s_pred.clip(lower=max_rate, upper=min_rate)

post_pred_charge_proc_func = lambda s_pred: (s_pred
                                      .pipe(clip_charge_rate)
                                      .pipe(normalise_total_charge)
                                     )

```

```python
post_pred_charge_proc_func(df_pred['pred']).groupby(s_pred.index.date).sum().value_counts()
```




    12.0    102
    12.0     60
    12.0     46
    12.0     23
    12.0     22
    12.0      1
    12.0      1
    Name: pred, dtype: int64



### Model comparison metrics

Schedules are scored according to the proportion of the total battery charge that comes from solar: $p_{d,1} = \frac{\sum{P_{d,k}}}{\sum{B_{d,k}}}$.

We will first write a function which evaluates this scoring function for a charging schedule and solar profile. 

```python
def score_charging(schedule, solar_profile):
    # The actual pv charge is the minimum of the scheduled charge and the actual solar availability 
    actual_pv_charge = np.minimum(schedule, solar_profile) 
    score = np.sum(actual_pv_charge)/np.sum(schedule)
    return score

# example: 
df_pred['pred'] = post_pred_charge_proc_func(df_pred['pred'])
schedule = sample_random_day(df_pred['pred'])
solar_profile = df.loc[schedule.index]['pv_power_mw']

print("Score for random day: {}".format(score_charging(schedule, solar_profile)))

# example: 
schedule = df_pred['pred']
solar_profile = df.loc[schedule.index]['pv_power_mw']
print("Score for entire dataset: {}".format(score_charging(schedule, solar_profile)))

```

    Score for random day: 0.9766702240622753
    Score for entire dataset: 0.7602308475908901
    

**However** remember that some days there is not enough solar PV to fill the battery. It would be good to know what % of the max score we achieved. That is, the sum of our PV charge over the total available PV capacity (capped at 6 MWh per day). 

```python
def max_available_solar(solar_profile, capacity_mwh=6, time_unit=0.5):
    """
    Return the solar PV potential available to the battery.
    
    That is, the total PV potential with a daily cap of 6 MWh. 
    """
    available = solar_profile.groupby(solar_profile.index.date).sum() * time_unit
    clipped = np.clip(available.values, 0, capacity_mwh)
    total = np.sum(clipped)
    return total 

max_available_solar(df.loc[schedule.index]['pv_power_mw'])    
```




    1283.5549999999998



Now we need a function to evaluate a schedule as a proportion of the max available score. That is, the total PV charge used by the battery divided by the total available solar PV. 

```python
def prop_max_solar(schedule, solar_profile, time_unit=0.5):
    actual_pv_charge = np.sum(np.minimum(schedule, solar_profile)*time_unit)
    max_pv_charge = max_available_solar(solar_profile)
    return actual_pv_charge/max_pv_charge

# example: 
df_pred['pred'] = post_pred_charge_proc_func(df_pred['pred'])
schedule = sample_random_day(df_pred['pred'])
solar_profile = df.loc[schedule.index]['pv_power_mw']
print("Score for random day: {}".format(prop_max_solar(schedule, solar_profile)))

# example: 
schedule = df_pred['pred']
solar_profile = df.loc[schedule.index]['pv_power_mw']
print("Score for entire dataset: {}".format(prop_max_solar(schedule, solar_profile)))
```

    Score for random day: 0.9669618605914381
    Score for entire dataset: 0.9244257572951786
    

### Model comparison

Now let's try some different models and view their scores and the proportion of maximum PV potential:

```python
models = {
    'std_linear': LinearRegression(),
    'random_forest': RandomForestRegressor(),
    'boosted': GradientBoostingRegressor()
}

for key in models:
    df_pred = clean.generate_kfold_preds(X, y, models[key], index=charging_datetimes)
    df_pred['pred'] = post_pred_charge_proc_func(df_pred['pred'])
    schedule = df_pred['pred']
    solar_profile = df.loc[schedule.index]['pv_power_mw']
    score = score_charging(schedule, solar_profile)
    prop_max = prop_max_solar(schedule, solar_profile)
    print("Model: `{}`    Score: {:.3f}     Proportion of max: {:.3f}%".format(key, 
                                                                              score,
                                                                              100*prop_max))
```

    Model: `std_linear`    Score: 0.760     Proportion of max: 90.629%
    Model: `random_forest`    Score: 0.771     Proportion of max: 91.890%
    Model: `boosted`    Score: 0.775     Proportion of max: 92.436%
    

```python
df_pred.groupby(df_pred.index.date).sum()
```




|            | pred   | true   |
|:-----------|:-------|:-------|
| 2017-11-10 | 12.0   | 12.0   |
| 2017-11-11 | 12.0   | 12.0   |
| 2017-11-12 | 12.0   | 12.0   |
| 2017-11-13 | 12.0   | 12.0   |
| 2017-11-14 | 12.0   | 12.0   |
| ...        | ...    | ...    |
| 2018-07-18 | 12.0   | 12.0   |
| 2018-07-19 | 12.0   | 12.0   |
| 2018-07-20 | 12.0   | 12.0   |
| 2018-07-21 | 12.0   | 12.0   |
| 2018-07-22 | 12.0   | 12.0   |</div>



<br>

Finally we'll export the relevant code to our `batopt` module
