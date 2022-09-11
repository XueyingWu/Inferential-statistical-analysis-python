
# Confidence Intervals


This tutorial is going to demonstrate how to load data, clean/manipulate a dataset, and construct a confidence interval for the difference between two population proportions and means.

We will use the 2015-2016 wave of the NHANES data for our analysis.

*Note: We have provided a notebook that includes more analysis, with examples of confidence intervals for one population proportions and means, in addition to the analysis I will show you in this tutorial.  I highly recommend checking it out!

For our population proportions, we will analyze the difference of proportion between female and male smokers.  The column that specifies smoker and non-smoker is "SMQ020" in our dataset.

For our population means, we will analyze the difference of mean of body mass index within our female and male populations.  The column that includes the body mass index value is "BMXBMI".

Additionally, the gender is specified in the column "RIAGENDR".


```python
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import seaborn as sns
%matplotlib inline
import matplotlib.pyplot as plt
import statsmodels.api as sm
```


```python
url = "nhanes_2015_2016.csv"
da = pd.read_csv(url)
```

### Investigating and Cleaning Data


```python
da.SMQ020.value_counts()
```




    2    3406
    1    2319
    9       8
    7       2
    Name: SMQ020, dtype: int64




```python
# Recode SMQ020 from 1/2 to Yes/No into new variable SMQ020x
da["SMQ020x"] = da.SMQ020.replace({1: "Yes", 2: "No", 7: np.nan, 9: np.nan})
da["SMQ020x"].value_counts()
```




    No     3406
    Yes    2319
    Name: SMQ020x, dtype: int64




```python
da.RIAGENDR.value_counts()
```




    2    2976
    1    2759
    Name: RIAGENDR, dtype: int64




```python
# Recode RIAGENDR from 1/2 to Male/Female into new variable RIAGENDRx
da["RIAGENDRx"] = da.RIAGENDR.replace({1: "Male", 2: "Female"})
da["RIAGENDRx"].value_counts()
```




    Female    2976
    Male      2759
    Name: RIAGENDRx, dtype: int64




```python
dx = da[["SMQ020x", "RIAGENDRx"]].dropna()
pd.crosstab(dx.SMQ020x, dx.RIAGENDRx)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th>RIAGENDRx</th>
      <th>Female</th>
      <th>Male</th>
    </tr>
    <tr>
      <th>SMQ020x</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>No</th>
      <td>2066</td>
      <td>1340</td>
    </tr>
    <tr>
      <th>Yes</th>
      <td>906</td>
      <td>1413</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Recode SMQ020x from Yes/No to 1/0 into existing variable SMQ020x
dx["SMQ020x"] = dx.SMQ020x.replace({"Yes": 1, "No": 0})
```


```python
dx.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>SMQ020x</th>
      <th>RIAGENDRx</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>Male</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>Male</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>Male</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0</td>
      <td>Female</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
      <td>Female</td>
    </tr>
  </tbody>
</table>
</div>




```python
dz = dx.groupby("RIAGENDRx").agg({"SMQ020x": [np.mean, np.size]})
dz.columns = ["Proportion", "Total n"]
dz
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Proportion</th>
      <th>Total n</th>
    </tr>
    <tr>
      <th>RIAGENDRx</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Female</th>
      <td>0.304845</td>
      <td>2972</td>
    </tr>
    <tr>
      <th>Male</th>
      <td>0.513258</td>
      <td>2753</td>
    </tr>
  </tbody>
</table>
</div>



### Constructing Confidence Intervals

Now that we have the population proportions of male and female smokers, we can begin to calculate confidence intervals.  From lecture, we know that the equation is as follows:

$$Best\ Estimate \pm Margin\ of\ Error$$

Where the *Best Estimate* is the **observed population proportion or mean** from the sample and the *Margin of Error* is the **t-multiplier**.

The equation to create a 95% confidence interval can also be shown as:

$$Population\ Proportion\ or\ Mean\ \pm (t-multiplier *\ Standard\ Error)$$

The Standard Error (SE) is calculated differenly for population proportion and mean:

$$Standard\ Error \ for\ Population\ Proportion = \sqrt{\frac{Population\ Proportion * (1 - Population\ Proportion)}{Number\ Of\ Observations}}$$

$$Standard\ Error \ for\ Mean = \frac{Standard\ Deviation}{\sqrt{Number\ Of\ Observations}}$$

Lastly, the standard error for difference of population proportions and means is:

$$Standard\ Error\ for\ Difference\ of\ Two\ Population\ Proportions\ Or\ Means = \sqrt{(SE_{\ 1})^2 + (SE_{\ 2})^2}$$

#### Difference of Two Population Proportions


```python
p = .304845
n = 2972
se_female = np.sqrt(p * (1 - p)/n)
se_female
```




    0.00844415041930423




```python
p = .513258
n = 2753
se_male = np.sqrt(p * (1 - p)/ n)
se_male
```




    0.009526078787008965




```python
se_diff = np.sqrt(se_female**2 + se_male**2)
se_diff
```




    0.012729880335656654




```python
d = .304845 - .513258
lcb = d - 1.96 * se_diff
ucb = d + 1.96 * se_diff
(lcb, ucb)
```




    (-0.23336356545788706, -0.18346243454211297)



#### Difference of Two Population Means


```python
da["BMXBMI"].head()
```




    0    27.8
    1    30.8
    2    28.8
    3    42.4
    4    20.3
    Name: BMXBMI, dtype: float64




```python
da.groupby("RIAGENDRx").agg({"BMXBMI": [np.mean, np.std, np.size]})
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead tr th {
        text-align: left;
    }

    .dataframe thead tr:last-of-type th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr>
      <th></th>
      <th colspan="3" halign="left">BMXBMI</th>
    </tr>
    <tr>
      <th></th>
      <th>mean</th>
      <th>std</th>
      <th>size</th>
    </tr>
    <tr>
      <th>RIAGENDRx</th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Female</th>
      <td>29.939946</td>
      <td>7.753319</td>
      <td>2976.0</td>
    </tr>
    <tr>
      <th>Male</th>
      <td>28.778072</td>
      <td>6.252568</td>
      <td>2759.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
sem_female = 7.753319 / np.sqrt(2976)
sem_male = 6.252568 / np.sqrt(2759)
(sem_female, sem_male)
```




    (0.14212523289878048, 0.11903716451870151)




```python
sem_diff = np.sqrt(sem_female**2 + sem_male**2)
sem_diff
```




    0.18538993598139303




```python
d = 29.939946 - 28.778072
```


```python
lcb = d - 1.96 * sem_diff
ucb = d + 1.96 * sem_diff
(lcb, ucb)
```




    (0.798509725476467, 1.5252382745235278)


