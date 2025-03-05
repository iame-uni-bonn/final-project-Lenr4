# Autoregressive Model Analysis and Multistep Forecast of Apple Stock Data

## Table of Contents

- [Overview](#overview)
- [System Prerequisites](#system-prerequisites)
- [Getting Started](#getting-started)
- [Project Structure](#project-structure)

______________________________________________________________________

## Overview

In this project, Apple stock data was analyzed using time series econometrics methods.
The final output of this project is the LaTeX file **paper.pdf** that roughly describes
the analysis steps as well as the figures/tables and deals with the question to what
extent it is possible to interpret the results.

In this project, I:

- fit several AR processes to Apple's historical stock data.
- compared model performance to identify the best fitting AR process.
- evaluated the ability of the best model to perform multistep forecasts.
- investigated the extent to which it is possible to use the AR process for analysis.
- provided analysis and plots to visualize both the model fit and forecasting
  performance.

______________________________________________________________________

## System Prerequisites

To make sure that the project works on your device it is necessary to have installed
*Python*, *a modern LaTeX distribution*, *Git*, and if applicable a *text editor*. For a
more detailed explanation see the
[documentation](https://econ-project-templates.readthedocs.io/en/stable/getting_started/index.html).

______________________________________________________________________

## Getting Started

First one needs to clone the repository:

```bash
git clone https://github.com/iame-uni-bonn/final-project-Lenr4.git
```

Next, navigate to the project root and create and activate the environment:

```bash
mamba env create lennart_epp
conda activate lennart_epp
```

After the environment is activated, one can run the project by:

```bash
pytask
```

> 🛑 **Caution**: If there were any trouble with kaleido on windows you need to use this
> [workaround](https://effective-programming-practices.vercel.app/plotting/why_plotly_prerequisites/objectives_materials.html#windows-workaround):
>
> ```bash
> pip install kaleido==0.1.0.post1
> ```

______________________________________________________________________

### Project Structure

The Project is structured into three different parts.

- **bld**: The Build directory cointaing all output files.

  - **plots**: Top 3 AR models for fitting (1 step forecast), ACF, Multistep forecast;
    all as interactive .html and .pdf files.
  - **forecasts**: Multistep forecast using AR(1) as .pkl file.
  - **data**: Cleaned Apple data as .pkl file.
  - **memory**: .pkl file of ACF and .tex files of *Hurst* and *ADF* statistics.
  - **models**: .pkl file of all AR models and .tex file with top model statistics.

- **src**: The Source directory containing all python files needed for the analysis.

  - **data**: CSV file containing the raw data for reproducibilty.
  - **data_management**: Python files for cleaning and downloading the data from
    [Yahoo Finance](https://de.finance.yahoo.com/).
  - **analysis**: Python files which analyse the data.
  - **final**: Python files which plot the results.

- **tests**: The Test directory containing all python files which are used for testing.

  - **data_management**: Python files for testing the data management steps.
  - **analysis**: Python files for testing the analysis steps.

______________________________________________________________________
