
# Hypothesis Testing

From lecture, we know that hypothesis testing is a critical tool in determing what the value of a parameter could be.

We know that the basis of our testing has two attributes:

**Null Hypothesis: $H_0$**

**Alternative Hypothesis: $H_a$**

The tests we have discussed in lecture are:

* One Population Proportion
* Difference in Population Proportions
* One Population Mean
* Difference in Population Means

In this tutorial, I will introduce some functions that are extremely useful when calculating a t-statistic and p-value for a hypothesis test.

Let's quickly review the following ways to calculate a test statistic for the tests listed above.

The equation is:

$$\frac{Best\ Estimate - Hypothesized\ Estimate}{Standard\ Error\ of\ Estimate}$$ 

We will use the examples from our lectures and use python functions to streamline our tests.


```python
import statsmodels.api as sm
import numpy as np
import pandas as pd
import scipy.stats.distributions as dist
```

### One Population Proportion

#### Research Question 

In previous years 52% of parents believed that electronics and social media was the cause of their teenager’s lack of sleep. Do more parents today believe that their teenager’s lack of sleep is caused due to electronics and social media? 

**Population**: Parents with a teenager (age 13-18)  
**Parameter of Interest**: p  
**Null Hypothesis:** p = 0.52  
**Alternative Hypthosis:** p > 0.52 (note that this is a one-sided test)

1018 Parents

56% believe that their teenager’s lack of sleep is caused due to electronics and social media.


```python
n = 1018
pnull = .52
phat = .56
#the amount of parents this it is true, n and pnull  
sm.stats.proportions_ztest(phat * n, n, pnull, alternative='larger', prop_var=0.52)
#returns zstat and p-value
```




    (2.571067795759113, 0.010138547731721065)



### Difference in Population Proportions

#### Research Question

Is there a significant difference between the population proportions of parents of black children and parents of Hispanic children who report that their child has had some swimming lessons?

**Populations**: All parents of black children age 6-18 and all parents of Hispanic children age 6-18  
**Parameter of Interest**: p1 - p2, where p1 = black and p2 = hispanic  
**Null Hypothesis:** p1 - p2 = 0  
**Alternative Hypthosis:** p1 - p2 $\neq$ = 0  


91 out of 247 (36.8%) sampled parents of black children report that their child has had some swimming lessons.

120 out of 308 (38.9%) sampled parents of Hispanic children report that their child has had some swimming lessons.


```python
# This example implements the analysis from the "Difference in Two Proportions" lecture videos

# Sample sizes
n1 = 247
n2 = 308

# Number of parents reporting that their child had some swimming lessons
y1 = 91
y2 = 120

# Estimates of the population proportions
p1 = round(y1 / n1, 2)
p2 = round(y2 / n2, 2)

# Estimate of the combined population proportion
phat = (y1 + y2) / (n1 + n2)

# Estimate of the variance of the combined population proportion
va = phat * (1 - phat)

# Estimate of the standard error of the combined population proportion
se = np.sqrt(va * (1 / n1 + 1 / n2))

# Test statistic and its p-value
test_stat = (p1 - p2) / se
pvalue = 2*dist.norm.cdf(-np.abs(test_stat))

# Print the test statistic its p-value
print("Test Statistic")
print(round(test_stat, 2))

print("\nP-Value")
print(round(pvalue, 2))
```

    Test Statistic
    -0.48
    
    P-Value
    0.63



```python
#An alternative way
n1=247
p1=.37

n2=308
p2=.39

population1 = np.random.binomial(1,p1,n1)
population2 = np.random.binomial(1,p2,n2)

sm.stats.ttest_ind(population1,population2)
#t statistic, p-value, total sample size
```




    (-0.4625385542801201, 0.6438771000936605, 553.0)



### One Population Mean

#### Research Question 

Is the average cartwheel distance (in inches) for adults 
more than 80 inches?

**Population**: All adults  
**Parameter of Interest**: $\mu$, population mean cartwheel distance.
**Null Hypothesis:** $\mu$ = 80
**Alternative Hypthosis:** $\mu$ > 80

25 Adults

$\mu = 82.46$

$\sigma = 15.06$


```python
df = pd.read_csv("Cartwheeldata.csv")
df.head()
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
      <th>ID</th>
      <th>Age</th>
      <th>Gender</th>
      <th>GenderGroup</th>
      <th>Glasses</th>
      <th>GlassesGroup</th>
      <th>Height</th>
      <th>Wingspan</th>
      <th>CWDistance</th>
      <th>Complete</th>
      <th>CompleteGroup</th>
      <th>Score</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>56</td>
      <td>F</td>
      <td>1</td>
      <td>Y</td>
      <td>1</td>
      <td>62.0</td>
      <td>61.0</td>
      <td>79</td>
      <td>Y</td>
      <td>1</td>
      <td>7</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>26</td>
      <td>F</td>
      <td>1</td>
      <td>Y</td>
      <td>1</td>
      <td>62.0</td>
      <td>60.0</td>
      <td>70</td>
      <td>Y</td>
      <td>1</td>
      <td>8</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>33</td>
      <td>F</td>
      <td>1</td>
      <td>Y</td>
      <td>1</td>
      <td>66.0</td>
      <td>64.0</td>
      <td>85</td>
      <td>Y</td>
      <td>1</td>
      <td>7</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>39</td>
      <td>F</td>
      <td>1</td>
      <td>N</td>
      <td>0</td>
      <td>64.0</td>
      <td>63.0</td>
      <td>87</td>
      <td>Y</td>
      <td>1</td>
      <td>10</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>27</td>
      <td>M</td>
      <td>2</td>
      <td>N</td>
      <td>0</td>
      <td>73.0</td>
      <td>75.0</td>
      <td>72</td>
      <td>N</td>
      <td>0</td>
      <td>4</td>
    </tr>
  </tbody>
</table>
</div>




```python
n = len(df)
mean = df["CWDistance"].mean()
sd = df["CWDistance"].std()
(n, mean, sd)
```




    (25, 82.48, 15.058552387264852)




```python
#value is value of null hypothesis
sm.stats.ztest(df["CWDistance"], value = 80, alternative = "larger")
```




    (0.8234523266982029, 0.20512540845395266)



### Difference in Population Means

#### Research Question 

Considering adults in the NHANES data, do males have a significantly higher mean Body Mass Index than females?

**Population**: Adults in the NHANES data.  
**Parameter of Interest**: $\mu_1 - \mu_2$, Body Mass Index.  
**Null Hypothesis:** $\mu_1 = \mu_2$  
**Alternative Hypthosis:** $\mu_1 \neq \mu_2$

2976 Females 
$\mu_1 = 29.94$  
$\sigma_1 = 7.75$  

2759 Male Adults  
$\mu_2 = 28.78$  
$\sigma_2 = 6.25$  

$\mu_1 - \mu_2 = 1.16$


```python
url = "nhanes_2015_2016.csv"
da = pd.read_csv(url)
da.head()
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
      <th>SEQN</th>
      <th>ALQ101</th>
      <th>ALQ110</th>
      <th>ALQ130</th>
      <th>SMQ020</th>
      <th>RIAGENDR</th>
      <th>RIDAGEYR</th>
      <th>RIDRETH1</th>
      <th>DMDCITZN</th>
      <th>DMDEDUC2</th>
      <th>...</th>
      <th>BPXSY2</th>
      <th>BPXDI2</th>
      <th>BMXWT</th>
      <th>BMXHT</th>
      <th>BMXBMI</th>
      <th>BMXLEG</th>
      <th>BMXARML</th>
      <th>BMXARMC</th>
      <th>BMXWAIST</th>
      <th>HIQ210</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>83732</td>
      <td>1.0</td>
      <td>NaN</td>
      <td>1.0</td>
      <td>1</td>
      <td>1</td>
      <td>62</td>
      <td>3</td>
      <td>1.0</td>
      <td>5.0</td>
      <td>...</td>
      <td>124.0</td>
      <td>64.0</td>
      <td>94.8</td>
      <td>184.5</td>
      <td>27.8</td>
      <td>43.3</td>
      <td>43.6</td>
      <td>35.9</td>
      <td>101.1</td>
      <td>2.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>83733</td>
      <td>1.0</td>
      <td>NaN</td>
      <td>6.0</td>
      <td>1</td>
      <td>1</td>
      <td>53</td>
      <td>3</td>
      <td>2.0</td>
      <td>3.0</td>
      <td>...</td>
      <td>140.0</td>
      <td>88.0</td>
      <td>90.4</td>
      <td>171.4</td>
      <td>30.8</td>
      <td>38.0</td>
      <td>40.0</td>
      <td>33.2</td>
      <td>107.9</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2</th>
      <td>83734</td>
      <td>1.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>1</td>
      <td>1</td>
      <td>78</td>
      <td>3</td>
      <td>1.0</td>
      <td>3.0</td>
      <td>...</td>
      <td>132.0</td>
      <td>44.0</td>
      <td>83.4</td>
      <td>170.1</td>
      <td>28.8</td>
      <td>35.6</td>
      <td>37.0</td>
      <td>31.0</td>
      <td>116.5</td>
      <td>2.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>83735</td>
      <td>2.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>2</td>
      <td>2</td>
      <td>56</td>
      <td>3</td>
      <td>1.0</td>
      <td>5.0</td>
      <td>...</td>
      <td>134.0</td>
      <td>68.0</td>
      <td>109.8</td>
      <td>160.9</td>
      <td>42.4</td>
      <td>38.5</td>
      <td>37.7</td>
      <td>38.3</td>
      <td>110.1</td>
      <td>2.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>83736</td>
      <td>2.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>2</td>
      <td>2</td>
      <td>42</td>
      <td>4</td>
      <td>1.0</td>
      <td>4.0</td>
      <td>...</td>
      <td>114.0</td>
      <td>54.0</td>
      <td>55.2</td>
      <td>164.9</td>
      <td>20.3</td>
      <td>37.4</td>
      <td>36.0</td>
      <td>27.2</td>
      <td>80.4</td>
      <td>2.0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 28 columns</p>
</div>




```python
females = da[da["RIAGENDR"] == 2]
male = da[da["RIAGENDR"] == 1]
```


```python
n1 = len(females)
mu1 = females["BMXBMI"].mean()
sd1 = females["BMXBMI"].std()

(n1, mu1, sd1)
```




    (2976, 29.93994565217392, 7.753318809545674)




```python
n2 = len(male)
mu2 = male["BMXBMI"].mean()
sd2 = male["BMXBMI"].std()

(n2, mu2, sd2)
```




    (2759, 28.778072111846942, 6.2525676168014614)




```python
sm.stats.ztest(females["BMXBMI"].dropna(), male["BMXBMI"].dropna())
```




    (6.1755933531383205, 6.591544431126401e-10)


