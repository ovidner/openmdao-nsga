# OpenMDAO-NSGA

[![DOI](https://zenodo.org/badge/DOI/10/f762.svg)](https://doi.org/f762)

NSGA-II and NSGA-III implementations for OpenMDAO.

Some notable features include:

* Support for continuous, integer, discrete and categorical variables.
* Support for Pareto-optimization of multi-objective problems.
* Constraint handling without parameters (using the method described in the original NSGA-II paper).
* Flexible termination criteria.

Do note that the performance for discrete problems will probably be bad, but if you are looking for a fairly robust *shotgun approach* for a variety of different problems, OpenMDAO-NSGA might serve you well enough.

## Installation

This assumes you are using Conda. Run this in your environment to install:

    conda install ovidner::openmdao-nsga

## Usage example

```python
import openmdao.api as om
import omnsga

prob = om.Problem()

...

# Real or integer design var
prob.model.add_design_var("x_1", lower=0, upper=1)
# Discrete design var
omnsga.add_design_var(prob.model, "x_2", values={0, 1, 4}, type=omnsga.VariableType.ORDINAL)
# Categorical design var
omnsga.add_design_var(prob.model, "x_3", values={True, False, None}, type=omnsga.VariableType.NOMINAL)

prob.model.add_constraint("g", lower=0, upper=1)
# Maximize f_1
prob.model.add_objective("f_1", scaler=-1)
# Minimize f_2
prob.model.add_objective("f_2")

prob.driver = omnsga.Nsga2Driver(
    termination_criterion=omnsga.MaxEvaluationsCriterion(500),
)

prob.run_driver()
```
