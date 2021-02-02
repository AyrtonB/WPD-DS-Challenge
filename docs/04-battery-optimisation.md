# Battery Optimisation



```python
#exports
import numpy as np
import pandas as pd
```

### Converting a charging schedule to capacity

The solution is given in terms of the battery charge/discharge schedule, but it is also necessary to satisfy constraints on the capacity of the battery (see below). 

The charge is determined by $C_{t+1} = C_{t} + 0.5B_{t}$

Note that we generally the initial value for capacity is 0 (the battery starts off empty). Change `init_value` if not.

```python
def charge_to_capacity(charge_schedule, init_value=0):
    capacity = np.append(np.array([init_value]), np.cumsum(charge_schedule[:-1]/2))
    return capacity
```

A simple example: 

```python
b = np.array([2.5, 0.5, 0, -1, 0])
charge_to_capacity(b)
```




    array([0.  , 1.25, 1.5 , 1.5 , 1.  ])



### Determine whether a solution meets constraints

The battery schedule must meet constraints such that all at timesteps the charging rate $B$ and capacity $C$ satisfy:

* $B_{min} \leq B \leq B_{max}$ 
* $0 \leq C \leq C_{max}$
 
In addition, we can only charge the battery between periods 1 (00:00) and 31 (15:00) inclusive, and discharge between periods 32 (15:30) and 42 (20:30) inclusive. For periods 43 to 48, there should be no activity, and the day must start with $C=0$. 


```python
B_min = -2.5 # Min rate of discharging
B_max = 2.5 # Max rate of charging
C_min = 0. # Min capacity
C_max = 6. # Max capacity 
charge_times = ('00:00', '15:00')
discharge_times = ('15:30', '20:30')
no_activity_times = ('21:00', '23:30')

def schedule_is_legal(schedule, B_min=B_min, B_max=B_max, C_min=C_min, C_max=C_max, 
                      charge_times=charge_times, discharge_times=discharge_times,
                      no_acivity_times=no_activity_times):
    """
    Determine if a battery schedule meets constraints
    """
    
    charge = schedule.charge_MW.values
    capacity = charge_to_capacity(schedule.charge_MW.values)
    schedule['capacity'] = capacity
    schedule = schedule.set_index(pd.to_datetime(schedule.datetime))
    
    if not np.all((charge >= B_min) & (charge <= B_max)): # charge constraints
        return False
    elif not np.all((capacity >= C_min) & (capacity <= C_max)): # capacity constraints
        return False
    elif np.any(schedule.between_time(discharge_times[0], discharge_times[1]).charge_MW.values > 0): # Discharge between discharge_times
        return False
    elif np.any(schedule.between_time(charge_times[0], charge_times[1]).charge_MW.values < 0): # Charge between charge_times 
        return False
    elif np.any(schedule.between_time(no_activity_times[0], no_activity_times[1]).charge_MW.values != 0): # No activity between no_activity_times 
        return False
    elif np.any(schedule.between_time('00:00', '00:00').capacity.values != 0): # Must be empty at 00:00
        return False
    else:
        return True
    
```

Testing out on a random schedule 

```python
example_schedule = pd.read_csv('../data/raw/example_valid_schedule.csv')

print(f"Is schedule legal? {schedule_is_legal(example_schedule)}")
```

    Is schedule legal? True
    

<br>

Finally we'll export the relevant code to our `batopt` module
