
## Hypothesis Tests in Python
In this assessment, you will look at data from a study on toddler sleep habits. 

The hypothesis tests you create and the questions you answer in this Jupyter notebook will be used to answer questions in the following graded assignment.


```python
import numpy as np
import pandas as pd
from scipy.stats import t
pd.set_option('display.max_columns', 30) # set so can see all columns of the DataFrame
```

Your goal is to analyse data which is the result of a study that examined
differences in a number of sleep variables between napping and non-napping toddlers. Some of these
sleep variables included: Bedtime (lights-off time in decimalized time), Night Sleep Onset Time (in
decimalized time), Wake Time (sleep end time in decimalized time), Night Sleep Duration (interval
between sleep onset and sleep end in minutes), and Total 24-Hour Sleep Duration (in minutes). Note:
[Decimalized time](https://en.wikipedia.org/wiki/Decimal_time) is the representation of the time of day using units which are decimally related.   


The 20 study participants were healthy, normally developing toddlers with no sleep or behavioral
problems. These children were categorized as napping or non-napping based upon parental report of
children’s habitual sleep patterns. Researchers then verified napping status with data from actigraphy (a
non-invasive method of monitoring human rest/activity cycles by wearing of a sensor on the wrist) and
sleep diaries during the 5 days before the study assessments were made.


You are specifically interested in the results for the Bedtime and Total 24-Hour Sleep Duration. 

Reference: Akacem LD, Simpkin CT, Carskadon MA, Wright KP Jr, Jenni OG, Achermann P, et al. (2015) The Timing of the Circadian Clock and Sleep Differ between Napping and Non-Napping Toddlers. PLoS ONE 10(4): e0125181. https://doi.org/10.1371/journal.pone.0125181


```python
# Import the data
df = pd.read_csv("nap_no_nap.csv") 
```


```python
# First, look at the DataFrame to get a sense of the data
df
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
      <th>id</th>
      <th>sex</th>
      <th>age (months)</th>
      <th>dlmo time</th>
      <th>days napped</th>
      <th>napping</th>
      <th>nap lights outl time</th>
      <th>nap sleep onset</th>
      <th>nap midsleep</th>
      <th>nap sleep offset</th>
      <th>nap wake time</th>
      <th>nap duration</th>
      <th>nap time in bed</th>
      <th>night bedtime</th>
      <th>night sleep onset</th>
      <th>sleep onset latency</th>
      <th>night midsleep time</th>
      <th>night wake time</th>
      <th>night sleep duration</th>
      <th>night time in bed</th>
      <th>24 h sleep duration</th>
      <th>bedtime phase difference</th>
      <th>sleep onset phase difference</th>
      <th>midsleep phase difference</th>
      <th>wake time phase difference</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>female</td>
      <td>33.7</td>
      <td>19.24</td>
      <td>0</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>20.45</td>
      <td>20.68</td>
      <td>0.23</td>
      <td>1.92</td>
      <td>7.17</td>
      <td>629.40</td>
      <td>643.00</td>
      <td>629.40</td>
      <td>-1.21</td>
      <td>-1.44</td>
      <td>6.68</td>
      <td>11.93</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>female</td>
      <td>31.5</td>
      <td>18.27</td>
      <td>0</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>19.23</td>
      <td>19.48</td>
      <td>0.25</td>
      <td>1.09</td>
      <td>6.69</td>
      <td>672.40</td>
      <td>700.40</td>
      <td>672.40</td>
      <td>-0.96</td>
      <td>-1.21</td>
      <td>6.82</td>
      <td>12.42</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>male</td>
      <td>31.9</td>
      <td>19.14</td>
      <td>0</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>19.60</td>
      <td>20.05</td>
      <td>0.45</td>
      <td>1.29</td>
      <td>6.53</td>
      <td>628.80</td>
      <td>682.60</td>
      <td>628.80</td>
      <td>-0.46</td>
      <td>-0.91</td>
      <td>6.15</td>
      <td>11.39</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>female</td>
      <td>31.6</td>
      <td>19.69</td>
      <td>0</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>19.46</td>
      <td>19.50</td>
      <td>0.05</td>
      <td>1.89</td>
      <td>8.28</td>
      <td>766.60</td>
      <td>784.00</td>
      <td>766.60</td>
      <td>0.23</td>
      <td>0.19</td>
      <td>6.20</td>
      <td>12.59</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>female</td>
      <td>33.0</td>
      <td>19.52</td>
      <td>0</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>19.21</td>
      <td>19.65</td>
      <td>0.45</td>
      <td>1.30</td>
      <td>6.95</td>
      <td>678.00</td>
      <td>718.00</td>
      <td>678.00</td>
      <td>0.31</td>
      <td>-0.13</td>
      <td>5.78</td>
      <td>11.43</td>
    </tr>
    <tr>
      <th>5</th>
      <td>6</td>
      <td>female</td>
      <td>36.2</td>
      <td>18.22</td>
      <td>4</td>
      <td>1</td>
      <td>14.00</td>
      <td>14.22</td>
      <td>15.00</td>
      <td>15.78</td>
      <td>16.28</td>
      <td>93.75</td>
      <td>137.00</td>
      <td>19.95</td>
      <td>20.25</td>
      <td>0.29</td>
      <td>1.26</td>
      <td>6.28</td>
      <td>602.20</td>
      <td>653.80</td>
      <td>695.95</td>
      <td>-1.73</td>
      <td>-2.03</td>
      <td>7.05</td>
      <td>12.06</td>
    </tr>
    <tr>
      <th>6</th>
      <td>7</td>
      <td>male</td>
      <td>36.3</td>
      <td>19.28</td>
      <td>1</td>
      <td>1</td>
      <td>14.75</td>
      <td>15.03</td>
      <td>15.92</td>
      <td>16.80</td>
      <td>16.08</td>
      <td>106.00</td>
      <td>80.00</td>
      <td>20.60</td>
      <td>20.96</td>
      <td>0.36</td>
      <td>2.12</td>
      <td>7.27</td>
      <td>618.40</td>
      <td>655.40</td>
      <td>724.40</td>
      <td>-1.32</td>
      <td>-1.68</td>
      <td>6.84</td>
      <td>11.99</td>
    </tr>
    <tr>
      <th>7</th>
      <td>8</td>
      <td>male</td>
      <td>30.0</td>
      <td>21.06</td>
      <td>5</td>
      <td>1</td>
      <td>13.09</td>
      <td>13.43</td>
      <td>14.44</td>
      <td>15.46</td>
      <td>15.82</td>
      <td>121.60</td>
      <td>163.80</td>
      <td>22.01</td>
      <td>22.53</td>
      <td>0.51</td>
      <td>2.92</td>
      <td>7.31</td>
      <td>526.80</td>
      <td>582.40</td>
      <td>648.40</td>
      <td>-0.95</td>
      <td>-1.47</td>
      <td>5.86</td>
      <td>10.25</td>
    </tr>
    <tr>
      <th>8</th>
      <td>9</td>
      <td>male</td>
      <td>33.2</td>
      <td>19.38</td>
      <td>2</td>
      <td>1</td>
      <td>14.41</td>
      <td>14.42</td>
      <td>15.71</td>
      <td>17.01</td>
      <td>16.60</td>
      <td>155.50</td>
      <td>131.25</td>
      <td>20.24</td>
      <td>20.37</td>
      <td>0.13</td>
      <td>1.60</td>
      <td>6.82</td>
      <td>626.80</td>
      <td>660.33</td>
      <td>782.30</td>
      <td>-0.86</td>
      <td>-0.99</td>
      <td>6.22</td>
      <td>11.44</td>
    </tr>
    <tr>
      <th>9</th>
      <td>10</td>
      <td>female</td>
      <td>37.1</td>
      <td>19.93</td>
      <td>3</td>
      <td>1</td>
      <td>13.12</td>
      <td>13.42</td>
      <td>14.31</td>
      <td>15.19</td>
      <td>15.30</td>
      <td>106.67</td>
      <td>130.67</td>
      <td>20.78</td>
      <td>21.63</td>
      <td>0.84</td>
      <td>2.20</td>
      <td>6.52</td>
      <td>549.50</td>
      <td>626.00</td>
      <td>656.17</td>
      <td>-0.76</td>
      <td>-1.82</td>
      <td>6.21</td>
      <td>10.59</td>
    </tr>
    <tr>
      <th>10</th>
      <td>11</td>
      <td>male</td>
      <td>32.9</td>
      <td>18.79</td>
      <td>4</td>
      <td>1</td>
      <td>13.99</td>
      <td>14.03</td>
      <td>14.85</td>
      <td>15.68</td>
      <td>16.10</td>
      <td>98.75</td>
      <td>126.60</td>
      <td>19.45</td>
      <td>19.88</td>
      <td>0.44</td>
      <td>1.34</td>
      <td>6.80</td>
      <td>655.20</td>
      <td>694.80</td>
      <td>753.95</td>
      <td>-0.66</td>
      <td>-1.09</td>
      <td>6.55</td>
      <td>12.01</td>
    </tr>
    <tr>
      <th>11</th>
      <td>12</td>
      <td>female</td>
      <td>35.0</td>
      <td>19.65</td>
      <td>5</td>
      <td>1</td>
      <td>13.18</td>
      <td>13.45</td>
      <td>14.33</td>
      <td>15.21</td>
      <td>15.35</td>
      <td>105.80</td>
      <td>130.40</td>
      <td>20.18</td>
      <td>20.84</td>
      <td>0.66</td>
      <td>1.93</td>
      <td>7.03</td>
      <td>611.20</td>
      <td>660.40</td>
      <td>717.00</td>
      <td>-0.53</td>
      <td>-1.19</td>
      <td>6.28</td>
      <td>11.38</td>
    </tr>
    <tr>
      <th>12</th>
      <td>13</td>
      <td>male</td>
      <td>35.1</td>
      <td>19.83</td>
      <td>3</td>
      <td>1</td>
      <td>13.94</td>
      <td>14.48</td>
      <td>15.26</td>
      <td>16.03</td>
      <td>15.78</td>
      <td>93.33</td>
      <td>110.20</td>
      <td>20.22</td>
      <td>20.89</td>
      <td>0.67</td>
      <td>1.99</td>
      <td>7.09</td>
      <td>611.80</td>
      <td>662.20</td>
      <td>705.13</td>
      <td>-0.39</td>
      <td>-1.06</td>
      <td>6.16</td>
      <td>11.26</td>
    </tr>
    <tr>
      <th>13</th>
      <td>14</td>
      <td>female</td>
      <td>35.6</td>
      <td>19.88</td>
      <td>4</td>
      <td>1</td>
      <td>12.68</td>
      <td>13.08</td>
      <td>13.92</td>
      <td>14.76</td>
      <td>15.00</td>
      <td>100.75</td>
      <td>139.33</td>
      <td>20.26</td>
      <td>20.80</td>
      <td>0.54</td>
      <td>1.96</td>
      <td>7.11</td>
      <td>618.80</td>
      <td>671.20</td>
      <td>719.55</td>
      <td>-0.38</td>
      <td>-0.92</td>
      <td>6.08</td>
      <td>11.23</td>
    </tr>
    <tr>
      <th>14</th>
      <td>15</td>
      <td>female</td>
      <td>36.6</td>
      <td>19.94</td>
      <td>4</td>
      <td>1</td>
      <td>12.71</td>
      <td>12.88</td>
      <td>13.80</td>
      <td>14.72</td>
      <td>14.88</td>
      <td>110.75</td>
      <td>130.00</td>
      <td>20.28</td>
      <td>20.92</td>
      <td>0.64</td>
      <td>1.49</td>
      <td>6.33</td>
      <td>548.00</td>
      <td>595.00</td>
      <td>658.75</td>
      <td>-0.34</td>
      <td>-0.90</td>
      <td>5.64</td>
      <td>10.39</td>
    </tr>
    <tr>
      <th>15</th>
      <td>16</td>
      <td>male</td>
      <td>36.5</td>
      <td>20.25</td>
      <td>3</td>
      <td>1</td>
      <td>13.74</td>
      <td>14.68</td>
      <td>15.66</td>
      <td>16.64</td>
      <td>16.45</td>
      <td>117.33</td>
      <td>162.75</td>
      <td>20.46</td>
      <td>21.25</td>
      <td>0.79</td>
      <td>2.19</td>
      <td>7.13</td>
      <td>593.25</td>
      <td>662.00</td>
      <td>710.58</td>
      <td>-0.21</td>
      <td>-1.00</td>
      <td>5.94</td>
      <td>10.88</td>
    </tr>
    <tr>
      <th>16</th>
      <td>17</td>
      <td>female</td>
      <td>33.7</td>
      <td>20.33</td>
      <td>5</td>
      <td>1</td>
      <td>13.15</td>
      <td>13.87</td>
      <td>14.49</td>
      <td>15.11</td>
      <td>15.40</td>
      <td>74.20</td>
      <td>135.00</td>
      <td>20.43</td>
      <td>21.03</td>
      <td>0.60</td>
      <td>2.44</td>
      <td>7.86</td>
      <td>649.80</td>
      <td>708.60</td>
      <td>724.00</td>
      <td>-0.10</td>
      <td>-0.70</td>
      <td>6.12</td>
      <td>11.53</td>
    </tr>
    <tr>
      <th>17</th>
      <td>18</td>
      <td>male</td>
      <td>36.4</td>
      <td>20.16</td>
      <td>5</td>
      <td>1</td>
      <td>12.47</td>
      <td>12.56</td>
      <td>13.30</td>
      <td>14.05</td>
      <td>14.25</td>
      <td>89.80</td>
      <td>107.00</td>
      <td>20.02</td>
      <td>20.45</td>
      <td>0.43</td>
      <td>1.23</td>
      <td>6.01</td>
      <td>573.60</td>
      <td>614.60</td>
      <td>663.40</td>
      <td>0.14</td>
      <td>-0.29</td>
      <td>5.07</td>
      <td>9.85</td>
    </tr>
    <tr>
      <th>18</th>
      <td>19</td>
      <td>female</td>
      <td>33.6</td>
      <td>19.68</td>
      <td>3</td>
      <td>1</td>
      <td>14.71</td>
      <td>14.85</td>
      <td>15.46</td>
      <td>16.07</td>
      <td>16.20</td>
      <td>73.00</td>
      <td>89.40</td>
      <td>19.50</td>
      <td>19.64</td>
      <td>0.14</td>
      <td>1.42</td>
      <td>7.20</td>
      <td>693.40</td>
      <td>715.00</td>
      <td>766.40</td>
      <td>0.18</td>
      <td>0.04</td>
      <td>5.74</td>
      <td>11.52</td>
    </tr>
    <tr>
      <th>19</th>
      <td>20</td>
      <td>male</td>
      <td>33.8</td>
      <td>20.51</td>
      <td>3</td>
      <td>1</td>
      <td>12.68</td>
      <td>13.54</td>
      <td>14.30</td>
      <td>15.07</td>
      <td>15.23</td>
      <td>91.67</td>
      <td>152.67</td>
      <td>20.18</td>
      <td>21.38</td>
      <td>1.19</td>
      <td>2.51</td>
      <td>7.63</td>
      <td>615.33</td>
      <td>692.00</td>
      <td>707.00</td>
      <td>0.33</td>
      <td>-0.87</td>
      <td>6.00</td>
      <td>11.12</td>
    </tr>
  </tbody>
</table>
</div>



**Question**: What value is used in the column 'napping' to indicate a toddler takes a nap? (see reference article) 

**Questions**: What is the overall sample size $n$? What are the sample sizes of napping and non-napping toddlers?

## Hypothesis tests
We will look at two hypothesis test, each with $\alpha = .05$:  


1. Is the average bedtime for toddlers who nap later than the average bedtime for toddlers who don't nap?


$$H_0: \mu_{nap}=\mu_{no\ nap}, \ H_a:\mu_{nap}>\mu_{no\ nap}$$
Or equivalently:
$$H_0: \mu_{nap}-\mu_{no\ nap}=0, \ H_a:\mu_{nap}-\mu_{no\ nap}>0$$


2. The average 24 h sleep duration (in minutes) for napping toddlers is different from toddlers who don't nap.


$$H_0: \mu_{nap}=\mu_{no\ nap}, \ H_a:\mu_{nap}\neq\mu_{no\ nap}$$
Or equivalently:
$$H_0: \mu_{nap}-\mu_{no\ nap}=0, \ H_a:\mu_{nap}-\mu_{no\ nap} \neq 0$$

First isolate `night bedtime` into two variables - one for toddlers who nap and one for toddlers who do not nap.


```python
nap_bedtime_0=df[df['napping']==1]
nap_bedtime=nap_bedtime_0['night bedtime']
```


```python
no_nap_bedtime_0 =df[df['napping']==0]
no_nap_bedtime=no_nap_bedtime_0['night bedtime']
```

Now find the sample mean bedtime for nap and no_nap.


```python
nap_mean_bedtime = nap_bedtime.mean()
nap_mean_bedtime
```




    20.304




```python
no_nap_mean_bedtime = no_nap_bedtime.mean()
no_nap_mean_bedtime
```




    19.590000000000003



**Question**: What is the sample difference of mean bedtime for nappers minus no nappers?


```python
mean_bedtime_diff = nap_mean_bedtime-no_nap_mean_bedtime
mean_bedtime_diff
```




    0.7139999999999951



Now find the sample standard deviation for $X_{nap}$ and $X_{no\ nap}$.


```python
# The np.std function can be used to find the standard deviation. The
# ddof parameter must be set to 1 to get the sample standard deviation.
# If it is not, you will be using the population standard deviation which
# is not the correct estimator
s1 = np.std(nap_bedtime,ddof=1) #nap_s_bedtime
```


```python
s2 = np.std(no_nap_bedtime,ddof=1) #no_nap_s_bedtime
```

**Question**: What is the s.e.$(\bar{X}_{nap} - \bar{X}_{no\ nap})$?

We expect the variance in sleep time for toddlers who nap and toddlers who don't nap to be the same. So we use a pooled standard error.

Calculate the pooled standard error of $\bar{X}_{nap} - \bar{X}_{no\ nap}$ using the formula below.

$s.e.(\bar{X}_{nap} - \bar{X}_{no\ nap}) = \sqrt{\frac{(n_1-1)s_1^2 + (n_2-1)s_2^2}{n_1+n_2-2}(\frac{1}{n_1}+\frac{1}{n_2})}$


```python
n1=len(nap_bedtime)
n2=len(no_nap_bedtime)
x=(n1-1)*s1*s1+(n2-1)*s2*s2
y=n1+n2-2
z=1/n1+1/n2
pooled_se =np.sqrt(x*z/y)
pooled_se
```




    0.2961871280370147



**Question**: Given our sample size of $n$, how many degrees of freedom ($df$) are there for the associated $t$ distribution?

Now calculate the $t$-test statistic for our first hypothesis test using  
* pooled s.e.($\bar{X}_{nap} - \bar{X}_{no\ nap}$)  
* $\bar{X}_{nap} - \bar{X}_{no\ nap}$  
* $\mu_{0,\ nap} - \mu_{0,\ no\ nap}=0$, the population difference in means under the null hypothesis


```python
tstat = (nap_mean_bedtime-no_nap_mean_bedtime)/pooled_se
tstat
```




    2.4106381824626966



**Question**: What is the p-value for the first hypothesis test?

For a discussion of probability density functions (PDF) and cumulative distribution functions (CDF) see:

https://integratedmlai.com/normal-distribution-an-introductory-guide-to-pdf-and-cdf/

To find the p-value, we can use the CDF for the t-distribution:
```
t.cdf(tstat, df)
```
Which for $X \sim t(df)$ returns $P(X \leq tstat)$.

Because of the symmetry of the $t$ distribution, we have that 
```
1 - t.cdf(tstat, df)
```
returns $P(X > tstat)$

The function `t.cdf(tstat, df)` will give you the same value as finding the one-tailed probability of `tstat` on a t-table with the specified degrees of freedom.

Use the function `t.cdf(tstat, df)` to find the p-value for the first hypothesis test.


```python
pvalue = 1-t.cdf(tstat, df)
pvalue
```




    0.013417041438843036




```python
a = nap_bedtime_0['24 h sleep duration']
b = no_nap_bedtime_0['24 h sleep duration']
```

**Question**: What are the t-statistic and p-value for the second hypothesis test?

Calculate the $t$ test statistics and corresponding p-value using the `scipy` function `scipy.stats.ttest_ind(a, b, equal_var=True)` and check with your answer. 

**Question**: Does `scipy.stats.ttest_ind` return values for a one-sided or two-sided test?

**Question**: Can you think of a way to recover the results you got using `1-t.cdf` from the p-value given by `scipy.stats.ttest_ind`?

Use the `scipy` function `scipy.stats.ttest_ind(a, b, equal_var=True)` to find the $t$ test statistic and corresponding p-value for the second hypothesis test.


```python
import scipy
from scipy import stats
scipy.stats.ttest_ind(a, b, equal_var=True)
```




    Ttest_indResult(statistic=1.4811248223284985, pvalue=0.1558664953018476)



**Question**: For the $\alpha=.05$, do you reject or fail to reject the first hypothesis?

**Question**: For the $\alpha=.05$, do you reject or fail to reject the second hypothesis?
