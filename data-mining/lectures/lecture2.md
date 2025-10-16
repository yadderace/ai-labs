# Getting to Know the Data

## Data
Data objects are described by variables. A variable V represents a property or characteristic of an object that may vary, either from one object to another or from one time to another.

### Variable Definition
A variable V is a quadruple <Name, Domain, Operations, Scale>:
* **Name**: The name of V
* **Domain**: The set of values of V
* **Operations**: The set of operations allowed over Domain
* **Scale**: A rule that associates a value from Domain for the variable V when it represents an object o.

## Types of Data
* Record Based Data (Transactions)
* Graph Based Data (WWW)
* Ordered Data (Genomics)

## Measuring Data
### Mean
Population Mean: $$\mu = \frac{1}{N} \sum_{i=1}^{N} x_i$$
Sample Mean: $$\bar{x} = \frac{1}{n} \sum_{i=1}^{n} x_i$$
Weighted Mean: $$\bar{x} = \frac{\sum_{i=1}^{n} w_i x_i}{\sum_{i=1}^{n} w_i}$$

### Median
Middle value if odd number of values, average of middle two values if even number of values

### Mode
Value that occurs most frequently

## Symmetric vs Skewed Data
* **Symmetric**: Data is symmetrically distributed around the mean. The mean, median, and mode are all equal.
* **Skewed**: Data is not symmetrically distributed around the mean.
  - **Left Skewed (Negative)**: Mean < Median < Mode
  - **Right Skewed (Positive)**: Mode < Median < Mean

## Dispersion
* **Range**: $\text{max}(x) - \text{min}(x)$
* **Quantile**: At most $n(k/q)$ values will be smaller
* **IQR**: $Q_3 - Q_1$ (middle 50% of data)
* **Outliers**: Values outside $[Q_1 - 1.5 \times IQR, Q_3 + 1.5 \times IQR]$
* **Variance**: $\sigma^2 = \frac{1}{N} \sum_{i=1}^{N} (x_i - \mu)^2$
* **Standard Deviation**: $\sigma = \sqrt{\sigma^2}$


## Data Characteristics
* **Dimensionality**: Number of variables/features. The curse of dimensionality refers to the exponential increase in data required to densely populate space as the dimension increases.
* **Sparsity**: Proportion of missing/zero values in the data.
* **Resolution**: Level of detail or aggregation in the data.

## Normal Distribution
* Bell-shaped curve
* $\mu = \bar{x}$
* $\sigma = \sqrt{\frac{1}{N} \sum_{i=1}^{N} (x_i - \mu)^2}$
* **68-95-99.7 Rule**:
  - 68% within $\mu \pm \sigma$
  - 95% within $\mu \pm 2\sigma$
  - 99.7% within $\mu \pm 3\sigma$

## Variable Types
* **Nominal**: Categories without order (gender, color, zip code)
* **Ordinal**: Ordered categories (education level, income level)
* **Interval**: Ordered with equal intervals (temperature in °C, dates)
* **Ratio**: Interval with true zero (height, weight, age)

## Statistical Plots
* **Boxplot**: Five-number summary (min, Q1, median, Q3, max) and outliers
* **Histogram**: Shows frequency distribution of numerical data
* **Quantile Plot**: Plots data against theoretical quantiles (index $f = \frac{i-0.5}{n}$)
* **Q-Q Plot**: Compares two distributions using their quantiles
* **Scatter Plot**: Shows relationship between two numerical variables

## Outlier Handling
1. Remove if erroneous
2. Transform (log, square root)
3. Use robust statistics (median, IQR)
4. Cap/floor extreme values
