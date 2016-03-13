# qq-pat

The qq-pat library provides you with an easy interface for the creation of graphs and the calculation of statistics for financial time series. The library uses time series stored within pandas dataframes to make its calculations. You can have either one or several columns within your dataframe and qq-pat will always calculate statistics for all of them and return the results as either a list (if you are calculating something like the Sharpe ratio) or an actual pandas dataframe with the same number of columns.

To use qq-pat is very easy. First make sure you create a pandas series or dataframe that contains either the daily returns or the daily price/balance for the assets or strategies you want to evaluate. After this simply initialize the qq-pat analyzer:

```
analyzer = qqpat.Analizer(data, column_type='price')
```

You can then use the analyzer object to create many different graphs or obtain the value for different statistics. For example if you want to obtain a PerformanceAnalytics style summary of system results you can use :

```
analyzer.plot_analysis_returns()
```

See the test.py file for a test showing you how to obtain different statistics of a group of three different stocks/ETFs. 

Use the "pydoc -w qqpat" command to generate an htm containing all function definitions available within the library plus relevant comments. 
