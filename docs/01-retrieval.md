# Data Retrieval



```python
#exports
import json
import numpy as np
import pandas as pd

import requests
```

```python
import FEAutils as hlp
import matplotlib.pyplot as plt
from IPython.display import JSON
```

<br>

### User Inputs

```python
raw_data_dir = '../data/raw'
intermediate_data_dir = '../data/intermediate'
```

<br>

### Public Holidays

We'll start by retrieving a JSON for public holidays available from www.gov.uk.

```python
get_holidays_json = lambda holidays_url='https://www.gov.uk/bank-holidays.json': requests.get(holidays_url).json()

holidays_json = get_holidays_json()

JSON(holidays_json)
```




    <IPython.core.display.JSON object>



<br>

We'll quickly save this data

```python
#exports
def save_latest_raw_holiday_data(raw_data_dir, holidays_url='https://www.gov.uk/bank-holidays.json'):
    holidays_json = get_holidays_json(holidays_url)
    
    with open(f'{raw_data_dir}/holidays.json', 'w') as fp:
        json.dump(holidays_json, fp)
        
    return 
```

```python
save_latest_raw_holiday_data(intermediate_data_dir)
```

<br>

We'll then convert it into a dataframe

```python
#exports
def load_holidays_df(raw_data_dir):
    with open(f'{raw_data_dir}/holidays.json', 'r') as fp:
        holidays_json = json.load(fp)

    df_holidays = pd.DataFrame(holidays_json['england-and-wales']['events'])
    df_holidays['date'] = pd.to_datetime(df_holidays['date'])
    
    return df_holidays
```

```python
df_holidays = load_holidays_df(raw_data_dir)

df_holidays.head()
```




|    | title                  | date       |   notes | bunting   |
|---:|:-----------------------|:-----------|--------:|:----------|
|  0 | New Year’s Day       | 2016-01-01 |     nan | True      |
|  1 | Good Friday            | 2016-03-25 |     nan | False     |
|  2 | Easter Monday          | 2016-03-28 |     nan | True      |
|  3 | Early May bank holiday | 2016-05-02 |     nan | True      |
|  4 | Spring bank holiday    | 2016-05-30 |     nan | True      |</div>



<br>

We'll now create a half-hourly time-series where the prescence of a public holiday is given a value of 1 

```python
#exports
def holidays_df_to_s(df_holidays):
    holidays_dt_range = pd.date_range(df_holidays['date'].min(), df_holidays['date'].max(), freq='30T', tz='UTC')

    s_holidays = pd.Series(np.isin(holidays_dt_range.date, df_holidays['date'].dt.date), index=holidays_dt_range).astype(int)
    s_holidays.index.name = 'datetime'
    s_holidays.name = 'holiday'
    
    return s_holidays
```

```python
s_holidays = holidays_df_to_s(df_holidays)

s_holidays.head()
```




    datetime
    2016-01-01 00:00:00+00:00    1
    2016-01-01 00:30:00+00:00    1
    2016-01-01 01:00:00+00:00    1
    2016-01-01 01:30:00+00:00    1
    2016-01-01 02:00:00+00:00    1
    Freq: 30T, Name: holiday, dtype: int32



<br>

We'll quickly plot the results

```python
fig, ax = plt.subplots(dpi=150)

s_holidays['2016'].plot()

hlp.hide_spines(ax, positions=['top', 'bottom', 'left', 'right'])
ax.set_yticks([])
ax.set_ylim(0.1, 0.9)
```




    (0.1, 0.9)




![png](img/nbs/output_16_1.png)


<br>

We'll create a wrapper for combining these steps

```python
#exports
def load_holidays_s(raw_data_dir):
    df_holidays = load_holidays_df(raw_data_dir)
    s_holidays = holidays_df_to_s(df_holidays)
    
    return s_holidays
```

```python
s_holidays = load_holidays_s(raw_data_dir)

s_holidays.head()
```




    datetime
    2016-01-01 00:00:00+00:00    1
    2016-01-01 00:30:00+00:00    1
    2016-01-01 01:00:00+00:00    1
    2016-01-01 01:30:00+00:00    1
    2016-01-01 02:00:00+00:00    1
    Freq: 30T, Name: holiday, dtype: int32



<br>

And also save the data to a csv

```python
s_holidays.to_csv(f'{intermediate_data_dir}/holidays.csv')
```

<br>

Finally we'll export the relevant code to our `batopt` module
