# WPD-DS-Challenge

This repository includes the workflow used by the UCL ESAIL team for submissions to the Western Power Distribition Data Science competition.

An example charging profile from our latest submission can be seen below

![submission_timeseries](img/latest_submission.png)

<br>
<br>

### Challenge Details

##### High-level Overview

* A 6MWh/2.5MW battery is connected to a primary distribution substation and a 5MW
solar farm in Devon, southwest England.
* Design the control of a storage device to support the distribution network to:
    * Maximise the daily evening peak reduction.
    * Using as much solar photovoltaic energy as possible.
* This will be done for each day for the week following the current challenge date.
* In other words it is a constrained optimisation/control problem under uncertainty.
* There will be four assessed weeks as part of this challenge.

A recording of the kick-off meeting can also be found [here](https://www.youtube.com/watch?t=1&v=Tu1bLROBNbo&feature=youtu.be&ab_channel=EnergySystemsCatapult).

<br>

##### Battery Charging

The aim of this compoennt is to maximise the proportion of the battery's charge that comes from solar PV. The total battery charge can be written as:

<img src="https://render.githubusercontent.com/render/math?math=B_{d,k} = P_{d,k} %2B G_{d,k}">

where <img src="https://render.githubusercontent.com/render/math?math=P_{d,k}"> is the power drawn to the battery from solar generation on day <img src="https://render.githubusercontent.com/render/math?math=d">, period <img src="https://render.githubusercontent.com/render/math?math=k"> , and <img src="https://render.githubusercontent.com/render/math?math=G_{d,k}"> is that drawn from the grid. 

Whenever the battery is charging, it will draw from solar PV as much as possible, and take the remainder from the grid. We can therefore express that for a period <img src="https://render.githubusercontent.com/render/math?math=k">, the battery will draw from PV an amount:

<img src="https://render.githubusercontent.com/render/math?math=P_k = \min(B_k, P_k^{Total})"> 

The remainder is drawn from the grid: 

<img src="https://render.githubusercontent.com/render/math?math=G_k = P_k - B_k">

The proportion of energy stored in the battery from solar PV on day <img src="https://render.githubusercontent.com/render/math?math=d"> is expressed as: 

<img src="https://render.githubusercontent.com/render/math?math=p_{d,1} = \frac{\sum_{k=1}^31 P_{d,k}}{\sum_{k=1}^31 B_{d,k}}">

An equivalent expression gives the proportion of energy stored in the battery that was drawn from the grid, <img src="https://render.githubusercontent.com/render/math?math=p_{d,2}">.

The scoring function is proportional to <img src="https://render.githubusercontent.com/render/math?math=3p_{d,1} %2B p_{d,2}">. We therefore want to maximise <img src="https://render.githubusercontent.com/render/math?math=p_{d,1}"> by coinciding our battery charging with the solar PV generation. Note that the minimum score that can be gained from this component is 1 (entirely charged from grid), and the maximum is 3 (entirely charged from solar PV).

<br>

##### Battery Discharging

We'll start by defining the cost function for the demand forecasting component of the battery discharge optimisation.

For each day (
<img src="https://render.githubusercontent.com/render/math?math=d"> = 1, … , 7) the peak percentage reduction is calculated using:

<img src="https://render.githubusercontent.com/render/math?math=R_{d, peak} = 100\left(\frac{\max_{k\in\{32,...,42\}}\left(L_{d, k}\right) - \max_{k\in\{32,...,42\}}\left(L_{d, k}+B_{d, k}\right)}{\max_{k\in\{32,...,42\}}\left(L_{d, k}\right)}\right)">

Where:
* <img src="https://render.githubusercontent.com/render/math?math=L_{d, k}"> is the average power (in MW) over the <img src="https://render.githubusercontent.com/render/math?math=k^{th}"> half hour of day <img src="https://render.githubusercontent.com/render/math?math=d">, where <img src="https://render.githubusercontent.com/render/math?math=k = 1"> would mean the period from midnight to 00:30 AM on the current day,  <img src="https://render.githubusercontent.com/render/math?math=d">. 
* <img src="https://render.githubusercontent.com/render/math?math=B_{d, k}"> is the average power (in MW) over the <img src="https://render.githubusercontent.com/render/math?math=k^{th}"> half hour of day <img src="https://render.githubusercontent.com/render/math?math=d">, to minimise the peak demand over the evening period (the half hours <img src="https://render.githubusercontent.com/render/math?math=k"> = 32 to 42)

Our goal is to maximise the peak percentage reduction from 3.30PM to 9PM.

<br>

##### Constraints

We also have a number of constraints. The first constraint is on the maximum import and export of energy, in this case:

<img src="https://render.githubusercontent.com/render/math?math=-2.5MW = B_{min} \leq B_{d, k} \leq B_{max} = 2.5MW">

Secondly the battery cannot charge beyond its capacity, <img src="https://render.githubusercontent.com/render/math?math=C_{d, k}">, (in MWh):

<img src="https://render.githubusercontent.com/render/math?math=0 \leq C_{d, k} \leq C_{max} = 6MWh">

The total charge in the battery at the next time step <img src="https://render.githubusercontent.com/render/math?math=C_{d, k+1}"> is related to how much is currently in the battery and how much charged within the battery at time <img src="https://render.githubusercontent.com/render/math?math=k">, i.e.

<img src="https://render.githubusercontent.com/render/math?math=C_{d, k+1} = C_{d, k} + 0.5B_{d, k}">

Finally, the battery must start empty at the start of each day in the test week. I.e. <img src="https://render.githubusercontent.com/render/math?math=C_{d,1} = 0"> for <img src="https://render.githubusercontent.com/render/math?math=d = 1,...,7">.

<br>
<br>

### Literature

The literature used in this work is being tracked using Zotero within the [ESAIL group](https://www.zotero.org/groups/2739875/esail/library), please add new papers and comment on existing ones. These should hopefully make it a lot easier down the line if we turn the work into a paper.

<br>
<br>

### Environment Set-Up

The easiest way to set-up your `conda` environment is with the `setup_env.bat` script for Windows. Alternatively you can carry out these manual steps from the terminal:

```bash
> conda env create -f environment.yml
> conda activate batopt
> ipython kernel install --user --name=batopt
```


<br>
<br>

### Nb-Dev Design Approach

##### What is Nb-Dev?

> `nbdev` is a library that allows you to develop a python library in Jupyter Notebooks, putting all your code, tests and documentation in one place. That is: you now have a true literate programming environment, as envisioned by Donald Knuth back in 1983!"

<br>

##### Why use Nb-Dev?

It enables notebooks to be used as the origin of both the documentation and the code-base, improving code-readability and fitting more nicely within the standard data-science workflow. The library also provides a [several tools](https://nbdev.fast.ai/merge.html) to handle common problems such as merge issues with notebooks.

<br>

##### How to use Nb-Dev?

Most of the complexity around `nbdev` is in the initial set-up which has already been carried out for this repository, leaving the main learning curve as the special commands used in notebooks for exporting code. The special commands all have a `#` prefix and are used at the top of a cell.

* `#default_exp <sub-module-name>` - the name of the sub-module that the notebook will be outputted to (put in the first cell)
* `#exports` - to export all contents in the cell
* `#hide` - to remove the cell from the documentation

These just describe what to do with the cells though, we have to run another function to carry out this conversion (which is normally added at the end of each notebook):

```python
from nbdev.export import notebook2script
    
notebook2script()
```
