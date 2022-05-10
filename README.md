# High-dimensional Linear Bandits with Variable Selection

# Author Contributions Checklist Form

## Data

### Abstract
We implement simulation and real data experiments, i.e., Africa soil property data and Riboflavin production data, in this paper. 

### Availability

1. Africa soil property data: training dataset in https://www.kaggle.com/c/afsis-soil-properties;
2. Riboflavin production data: provided in the supplementary material (riboflavin.csv) of Bühlmann et al. (2014). (https://www.annualreviews.org/doi/abs/10.1146/annurev-statistics-022513-115545)

### Description

We provide the data file:
1. Africa soil data: `trainging.csv` contains the training dataset from (https://www.kaggle.com/competitions/afsis-soil-properties/data), including response variable `Ca` and covariate `Depth` that is used to partition the dataset into two groups for bandit experiment.
2. Riboflavin data: `riboflavin.csv` contains the dataset https://www.annualreviews.org/doi/abs/10.1146/annurev-statistics-022513-115545.

## Code

### Abstract

All of the simulation and real data experiments are done in R. We provide R code that is necessary for the replication of our empirical results. 

### Description

#### Version of primary software used

R version 4.0.1

#### Libraries and dependencies used by the code

R packages: glmnet; MASS; ncvreg; picasso; foreach; doParallel

#### Parallelization used

Multi-core parallelization on a single machine/node: 22 cores used.

## Instructions for use

### Reproducibility

- The *simulation* folder contains the R code for reproducing simulation results in the paper corresponding to $K=5, d=100, s_0=5$ scenario. The other two scenarios ($d=1000$ and $d=20$) can be reproduced by changing parameters in the R code. Below is the description of R code files for all compared methods.
    - `mcp_ucb_vs_cv.R`: our proposed UCB-VS algorithm with MCP 
    - `scad_ucb_vs_cv.R`: our proposed UCB-VS algorithm with SCAD
    - `mcp_ucb_cv.R`: $\ell_1$-confidence ball based algorithm with MCP
    - `scad_ucb_cv.R`: $\ell_1$-confidence ball based algorithm with SCAD
- The *real_data* folder contains the R-code for two real data experiments in the paper.
- *africa_soil* folder contains R-code for Africa soil property data.
    - `trainging.csv`: the file contains the training dataset from (https://www.kaggle.com/competitions/afsis-soil-properties/data), including response variable `Ca` and covariate `Depth` that is used to partition the dataset into two groups for bandit experiment.
    - `mcp_ucb_vs_ridge.R`: UCB-VS algorithm with MCP and $\alpha=1$ for additional $\ell_2$-penalty.
    - `mcp_ucb_vs_ridge_cv.R`: UCB-VS algorithm with MCP and cross-validation.
    - `mcp_ucb_ridge.R`: $\ell_1$-confidence ball based algorithm with MCP and $\alpha=1$ for additional $\ell_2$-penalty.
    - `scad_ucb_vs_ridge.R`: UCB-VS algorithm with SCAD and $\alpha=1$ for additional $\ell_2$-penalty.
    - `scad_ucb_vs_ridge_cv.R`: UCB-VS algorithm with SCAD and cross-validation.
    - `scad_ucb_ridge.R`: $\ell_1$-confidence ball based algorithm with SCAD and $\alpha=1$ for additional $\ell_2$-penalty.
    - Other $\alpha$ values (e.g. $1$ and $1/9$) for $\ell_2$-penalty can also be set. More details can be found from the comments in R code.
- *riboflavin* folder contains R-code for Riboflavin production data.
    - `riboflavin.csv`: the file contains the dataset
