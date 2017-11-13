import pandas as pd
from pandas_datareader import data
import datetime
import qqpat

aapl = data.DataReader('AAPL','yahoo',start=datetime.datetime(2016, 1, 1),end=datetime.datetime.now()) 
                                 
spy = data.DataReader('SPY','yahoo',start=datetime.datetime(2016, 1, 1),end=datetime.datetime.now()) 
                                 
ibm = data.DataReader('IBM','yahoo',start=datetime.datetime(2016, 1, 1),end=datetime.datetime.now())                                

data = pd.concat([aapl['Close'], spy['Close'], ibm['Close']], axis=1)

analyzer = qqpat.Analizer(data, column_type='price', titles=["APPL", "SPY", "IBM"])

summary = analyzer.get_statistics_summary()

for idx, statistics in enumerate(summary):
    print ""
    print "statistics for system {}:".format(idx)
    for s in statistics:
        print "{}: {}".format(s, summary[idx][s])
    print ""
    
analyzer.plot_analysis_returns()
analyzer.min_variance_portfolio_optimization(plotWeights=True)

analyzer.plot_mc_wc_evolution_sharpe(index=0, iterations=50, confidence=99, max_period_length=1000)
analyzer.plot_mc_wc_evolution_cagr(index=0, iterations=50, confidence=99, max_period_length=1000)

analyzer.plot_mc_distributions(index=0, iterations=100)
analyzer.plot_mc_simulations(index=0, iterations=100)

analyzer.plot_analysis_rolling()
analyzer.plot_monthly_returns_heatmap()
analyzer.plot_annual_returns()
analyzer.plot_monthly_returns()
analyzer.plot_annual_returns()
analyzer.plot_monthly_return_distribution()

analyzer.plot_drawdown_periods()
analyzer.plot_drawdown_distribution()
analyzer.plot_drawdown_length_distribution()



