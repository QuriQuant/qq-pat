import pandas as pd
from pandas_datareader import data
import datetime
import qqpat

aapl = data.get_data_yahoo('AAPL',
                                 start=datetime.datetime(2003, 10, 1),
                                 end=datetime.datetime(2015, 1, 1))
                                 
spy = data.get_data_yahoo('TLT',
                                 start=datetime.datetime(2003, 10, 1),
                                 end=datetime.datetime(2015, 1, 1))
                                 
ibm = data.get_data_yahoo('IBM',
                                 start=datetime.datetime(2003, 10, 1),
                                 end=datetime.datetime(2015, 1, 1))                                 

data = pd.concat([aapl['Adj Close'], spy['Adj Close'], ibm['Adj Close']], axis=1)

analyzer = qqpat.Analizer(data, column_type='price', titles=["APPL", "TLT", "IBM"])

summary = analyzer.get_statistics_summary()

for idx, statistics in enumerate(summary):
    print ""
    print "statistics for system {}: {}".format(idx, statistics)
    print ""

analyzer.min_variance_portfolio_optimization(plotWeights=True)

print analyzer.get_mc_statistics(index=0, iterations=200, confidence=99)

analyzer.plot_mc_wc_evolution_sharpe(index=0, iterations=50, confidence=99, max_period_length=1000)
analyzer.plot_mc_wc_evolution_cagr(index=0, iterations=50, confidence=99, max_period_length=1000)

analyzer.plot_mc_distributions(index=0, iterations=100)
analyzer.plot_mc_simulations(index=0, iterations=100)

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


