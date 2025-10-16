# Data Preprocessing

## Overview
The detection and correction of the data quality
problems. The use of algorithms that can tolerate poor data
quality.
Data cab be:
- Inconsistent: data transformations, technology problems, human errors
- Incomplete: missing values, incomplete records
- Inaccurate: errors in data entry, data transformations, technology problems
- Outdated: data transformations, technology problems

### Data Quality

## Data Cleaning 
Converting data so that it becomes consistent, complete, accurate, and up-to-date.
It's realied by filling missing values, removing duplicates, smoothing noise, resolving inconsistencies.

How to handle noisy data:
- Clustering (detect and remove outliers)
- Computer and human inspection

How to handle missing data:
- Filling manually
- Using the variable mode, median or mean.

## Data Integration
Combining data from different sources.

### Possible problems:
- Different variables have the same name
- Simmilar variables have different name.
- Redundant variable: can be detected wit Chi-Square, Covariance analysis.

Chi-square test:
$$\chi^2 = \sum_{i=1}^{n} \frac{(O_i - E_i)^2}{E_i}$$
Covariance:
$$\text{Cov}(X, Y) = \frac{1}{N} \sum_{i=1}^{N} (x_i - \mu_x)(y_i - \mu_y)$$


## Data Reduction
Obtaining a reduced set of variables that are sufficient to represent the data.
Strategies:
- Principal Component Analysis (PCA): converts variables into a new set of variables that are uncorrelated and capture the maximum variance.
- Multidimensional Scaling (MDS): Find a low-dimensional representation that preserves pairwise distances (or dissimilarities) between points.
- Feature selection: select a subset of variables that are most relevant to the task.
- Clustering: group similar objects together.
- Sampling: is the main strategy for data reduction in data mining. The sample must be representative of the population. There are some types of sampling:
  - Without replacement: each object is selected only once.
  - With replacement: each object can be selected multiple times.
  - Stratified sampling: data is split into partitions and a sample is taken from each partition.

### Data Valuation
Seeks to assign a numerical value to an individual’s data in the trade of data.
The issue is the time and cost of data valuation. Complexity is above $O(2^N)$.

## Data Transformation and Discretization
A function that maps the entire set of values of a given variable to a new set of replacement values.
Methods:
- Normalization: scales the values to a range, such as [0, 1] or [-1, 1].
- Smoothing: reduces noise in the data.
- Variable / Feature construction: creates new variables from existing variables.

### Normalization:
- Min-Max: scales the values to a range, such as [0, 1] or [-1, 1]. [New Min, New Max]
$$\bar{x} = \frac{x - \min(x)}{\max(x) - \min(x)}$$
- Z-Score: scales the values to have a mean of 0 and a standard deviation of 1.
$$\bar{x} = \frac{x - \mu}{\sigma}$$

### Discretization:
Divides the range of continuos values into a set of intervals. The intervals are called bins and can replace the original values. Also clustering can be used to find the intervals.
- Binning: we can do it with equal width or equal frequency (depth).