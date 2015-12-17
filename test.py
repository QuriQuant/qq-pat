import pandas as pd
from pandas_datareader import data
import datetime
import qqpat

aapl = data.get_data_yahoo('AAPL',
                                 start=datetime.datetime(2000, 10, 1),
                                 end=datetime.datetime(2015, 1, 1))
                                 
spy = data.get_data_yahoo('SPY',
                                 start=datetime.datetime(2000, 10, 1),
                                 end=datetime.datetime(2015, 1, 1))
                                 
ibm = data.get_data_yahoo('IBM',
                                 start=datetime.datetime(2000, 10, 1),
                                 end=datetime.datetime(2015, 1, 1))                                 

data = pd.concat([aapl['Adj Close'], spy['Adj Close'], ibm['Adj Close']], axis=1)

analyzer = qqpat.Analizer(data, column_type='price', titles=["appl", "spy", "ibm"])

print "Monte carlo statistics using 200 simulations for the first symbol"
print analyzer.get_mc_statistics(index=0, iterations=200, confidence=99)

analyzer.plot_analysis_returns()
analyzer.plot_analysis_rolling()
analyzer.plot_monthly_returns_heatmap()
analyzer.plot_annual_returns()
analyzer.plot_monthly_returns()
analyzer.plot_annual_returns()
analyzer.plot_monthly_return_distribution()
analyzer.plot_drawdown_distribution()
analyzer.plot_drawdown_length_distribution()
analyzer.plot_drawdown_periods()


