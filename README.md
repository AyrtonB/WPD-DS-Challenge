# WPD-DS-Challenge

This repository includes the workflow used by the UCL ESAIL team for submissions to the Western Power Distribition Data Science competition.

<br>
<br>

### Challenge Details

High-level overview:

* A 6MWh/2.5MW battery is connected to a primary distribution substation and a 5MW
solar farm in Devon, southwest England.
* Design the control of a storage device to support the distribution network to:
    * Maximise the daily evening peak reduction.
    * Using as much solar photovoltaic energy as possible.
* This will be done for each day for the week following the current challenge date.
* In other words it is a constrained optimisation/control problem under uncertainty.
* There will be four assessed weeks as part of this challenge.

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

### Nb-Dev

##### What is Nb-Dev?

> `nbdev` is a library that allows you to develop a python library in Jupyter Notebooks, putting all your code, tests and documentation in one place. That is: you now have a true literate programming environment, as envisioned by Donald Knuth back in 1983!"

<br>

##### Why use Nb-Dev?

It enables notebooks to be used as the origin of both the documentation and the code-base, improving code-readability and fitting more nicely within the standard data-science workflow

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