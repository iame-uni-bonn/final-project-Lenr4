# Apple Stock AR-Process Analysis & Multistep Forecasting

## Table of Contents

- [Overview](#overview)
- [System Prerequisites](#system-prerequisites)
- [Getting Started](#getting-started)
- [Project Structure](#project-structure)

______________________________________________________________________

## Overview

In this project, I analyzed apple stock data using time series econometrics methods. The
final result of my project is a latex file that roughly describes the analysis steps and
the charts and to what extent it is possible to use/interpret my results.

In this project, I:

- fit several AR processes to Apple's historical stock data.
- compare model performance to identify the best fitting AR process.
- evaluate the ability of the best model to perform multistep forecasts.
- investigate the extent to which it is possible to use the AR process for analysis.
- provide analysis and plots to visualize both the model fit and forecasting
  performance.

______________________________________________________________________

## System Prerequisites

To make sure that the project works on your machine you need to have installed *Python*,
*a modern LaTeX distribution*, *Git*, and if applicable a *text editor*. For a more
detailed explanation see the
[documentation](https://econ-project-templates.readthedocs.io/en/stable/getting_started/index.html).

______________________________________________________________________

## Getting Started

First one needs to clone the repository:

```bash
git clone https://github.com/iame-uni-bonn/final-project-Lenr4.git
```

Next navigate to the project root and create and activate the environment:

```bash
mamba env create lennart_epp
conda activate lennart_epp
```

After the environment is activated, one can run the project by:

```bash
pytask
```

> 🛑 **Caution**: If you had trouble with kaleido on windows you need to use this
> [workaround](https://effective-programming-practices.vercel.app/plotting/why_plotly_prerequisites/objectives_materials.html#windows-workaround):
>
> ```bash
> pip install kaleido==0.1.0.post1
> ```

______________________________________________________________________

### Project Structure

The Project is structured into three different parts.

- **bld**: The Build directory cointaing all output files.

  - **plots**: top 3 AR models for fitting(1 step forecast), multistep forecast, ACF all
    as interactive html and pdf
  - **forecasts**: 10 step forecast using AR(1) as pkl file
  - **data**: cleaned apple data as pkl file
  - **memory**: pkl file of ACF, and tex files of *Hurst* and *ADF* statistics
  - **models** pkl file of all AR models and tex file with top model statistics

- **src**: The source directory containing all python files needed for the analysis.

  - **data**: CSV file containing the raw data for reproducibilty.
  - **data_management**: Python files for cleaning and downloading the data from
    [Yahoo Finance](https://de.finance.yahoo.com/).
  - **analysis**: Python files which analyse the data.
  - **final**: Python files which plot the results.

- **tests**: The test directory containing all python files which are used for testing.

  - **data_management**: Python files for testing the data management steps.
  - **analysis**: Python files for testing the analysis steps.

______________________________________________________________________
