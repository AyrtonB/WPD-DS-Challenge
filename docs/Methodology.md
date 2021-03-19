# Method Outline

<br>

The following questions will help us understand the different approaches which were taken during the challenge as well as the effect of different inputs/features. Please limit total responses to no more than 2 pages.

<br>

### High-Level Overview

> Please describe a high-level overview of the methods you applied to solve the challenge tasks? If you solved the problem into different components (for example were the discharging and charging components solved separately?) what methods were applied to solve each part?

Given the competition assumptions, as long as the battery is fully charged and discharged each day there are no additional benefits to modelling the charge/discharge periods in combination (whereas this does increase the complexity of the task). For this reason the discharging and charging profiles were optimised for separately in this work. There was also a practical element as the two team members were able to split the work more easily in this way.

In the first submission both the charge and discharge optimisations were treated as a supervised learning problem, with the "perfect" charge/discharge profile calculated for the historical data and then used as the dependant variable. Due to the higher uncertainty of the solar time-series (relative to demand) we discovered that it was beneficial to forecast the solar output first, then apply our peak flattening algorithm to the forecasted output. For the discharge model the supervised learning approach (with discharge as the dependent variable) was found to out-perform the two-stage forecast and flatten method.

We used Gaussian Process Regression to model the hyper-parameter optimisation surface in order to carry out a more 'intelligent' search of the parameter space. We used sequential feature selection to select the variables used as inputs. For both the hyper-parameter optimisation and feature selection the metric used was the same as that in the submission evaluation.

<br>

### Software

> What is the name of the software (proprietary or open-source) which were primarily used to solve the challenge?

We used Python, making extensive use of the `pandas`, `sklearn`, and `skopt` libraries. `dagster` was used to generate an end-to-end pipeline from data retrieval to charge/discharge scheduling.

<br>

### Model Inputs

> What inputs did you include within the models? Please also include whether you used lagged versions of the variables, any interacting variables, or any features generated/engineered from the data.

Discharge Model (in order of importance):

* hour (integer for time of day)                
* doy (integer for day of year)                      
* temp_location4          
* weekend (dummy variable)             
* SP_demand_7d_lag (demand from a week prior)      
* evening_demand_max_7d_lag (max evening demand from a week prior)
* evening_demand_avg_7d_lag (mean evening demand from a week prior)
* temp_location3               
* daily_avg_temp (average over the full day)            
* temp_location2               
* spatial_avg_temp       
* temp_location1               
* temp_location6               
* temp_location5               
* dow (integer for day of week)  

As the main model used was a Random Forest (which can handle discrete changes) the temporal variables did not need further feature engineering.

<br>

PV Model (no specific order):

* temp_location1
* temp_location2
* temp_location3
* temp_location4
* temp_location5
* temp_location6
* solar_location1
* solar_location2
* solar_location3
* solar_location5
* solar_location6
* pv_7d_lag
* hour
* doy

<br>

### Model Evolution

> What changes did you make to your method(s) as the tasks progressed? What were some of the reasonings behind these changes?

The key change was the previously discussed move from a supervised learning model for the charge time-series to instead forecasting solar and then flattening the forecast peak. Beyond this we also adjusted our data cleaning method due to the varying issues as new data was released. For each submission we also re-tuned the hyper-parameters and explored the effect of adjusting the coverage of the training data.

<br>

### Christmas Quirks

> Did you treat the Christmas period task (Task 3) differently compared to the other tasks? If so, what changes did you make?

We carried out EDA to determine whether to switch to a similar days/week method for the Christmas submission instead of the previous supervised learning approach. The resulting output was almost identical to the supervised learning approach so we opted to keep the same methodology as used in other weeks for consistency.

<br>

### Covid Quirks

> Did you treat the task occurring during the COVID lock-down period (Task 4) differently to the other tasks? If so, what changes did you make?

The Covid variables we included didn't have a significant impact but included: a covid indicator (0 before 26th march (first lockdown date), 1 afterwards) and the number of days since 26th March 2020. The latter variable was capped at 15th June, when there was a significant easing of restrictions.

<br>

### Data Coverage

> How much of the data did you use for training, testing and (where applicable) model selection/validation?

Prior to the Covid period submission all data was used, for this final submission though the discharge model was trained only on data generated since March 2020. For model selection/validation we used k-fold validation where all days in a given week were grouped together (i.e. not split over multiple batches).

<br>

### General Comments

> Any other comments or suggestions about the challenge?

Our team really enjoyed the challenge, in particular the focus on real-world constraints. It would have been great if the carbon emissions of the grid were based on real-world data from somewhere like www.carbonintensity.org.uk. Although difficult to address we did wonder if there were potential options to normalise the weekly scores to counter-act their varying contribution in the combined submissions score.