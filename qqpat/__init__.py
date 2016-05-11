from cvxpy import *
import datetime
import pandas as pd
from pandas import Series, DataFrame
import numpy as np
from math import *
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
from matplotlib.dates import DateFormatter
import matplotlib as mpl
import matplotlib.mlab as mlab
import seaborn as sns
from random import randint
from sklearn import covariance

__version__                = "1.522"
ROLLING_PLOT_PERIOD        = 12

SAMPLE_COVARIANCE          = 0
LEDOIT_WOLF                = 1
OAS                        = 2
SHRUNK_SAMPLE_COVARIANCE   = 3


def lastValue(x):
    try:
        reply = x[-1]
    except:
        reply = None
    return reply
    

class Analizer:

    """
    Main class used for the analysis of financial time series 
    """

    def __init__(self, df, column_type='return', titles=None):
    
        """
        Initialization function creates the analyzer object used by class methods
        it takes a pandas dataframe, and optional column_type and titles variables.
        The column_type can be "return" or "price" depending on whether the 
        input dataframe contains prices or returns. Titles contains a list of strings
        to be assigned as dataframe titles.
        """
        
        if column_type == 'price':    
            all_series = []
            df = pd.DataFrame(df)
            for i in range(0, len(df.columns)):
                series = pd.Series(df.ix[:,i])
                returns = series.pct_change(fill_method='pad')
                all_series.append(returns)
            self.data = pd.concat(all_series, axis=1) 
            self.data = self.data.fillna(0.0)            
        elif column_type == 'return':
            self.data = pd.DataFrame(df)
            self.data = self.data.fillna(0.0)
        else:
            raise ValueError('column_type \'{}\' not valid.'.format(column_type))
            
        self.use_titles = False
        
        if titles != None:
            if len(titles) == len(self.data.columns):
                self.data.columns = titles
                self.use_titles = True
            else:
                raise ValueError('Number of titles is different from number of columns in data series.')    

        self.statistics = {}
        self.series = {}
        
    def get_statistics_summary(self, input_df = None, external_df = False):
    
        """
        Returns a dictionary containing a summary of all basic statistics for all 
        the columns within the input dataframe.
        """
    
        if 'summary' in self.statistics and external_df == False:
            return self.statistics['summary']
    
        all_win_ratio               = self.get_win_ratio(input_df, external_df) #
        all_reward_to_risk          = self.get_reward_to_risk(input_df, external_df)
        all_cagr                    = self.get_cagr(input_df, external_df)
        all_sharpe_ratio            = self.get_sharpe_ratio(input_df, external_df)      
        all_mar_ratio               = self.get_mar_ratio(input_df, external_df)
        all_average_return          = self.get_returns_avg(input_df, external_df)
        all_stddev_return           = self.get_returns_std(input_df, external_df) 
        all_average_dd              = self.get_average_dd(input_df, external_df)
        all_profit_factor           = self.get_profit_factor(input_df, external_df)
        all_pearson_correlation     = self.get_pearson_correlation(input_df, external_df)
        all_max_drawdown            = self.get_max_dd(input_df, external_df)
        all_average_dd_length       = self.get_average_dd_length(input_df, external_df)
        all_longest_dd_period       = self.get_longest_dd_period(input_df, external_df)
        all_average_recovery        = self.get_average_recovery(input_df, external_df)
        all_longest_recovery        = self.get_longest_recovery(input_df, external_df)
        all_burke_ratio             = self.get_burke_ratio(input_df, external_df)
        all_ulcer_index             = self.get_ulcer_index(input_df, external_df)
        all_martin_ratio            = self.get_martin_ratio(input_df, external_df)
        all_sortino_ratio           = self.get_sortino_ratio(0.0, input_df, external_df)
        
        all_statistics = []
        
        for i in range(0, len(self.data.columns)):
        
            statistics = {}                    
            statistics['win ratio']                 = all_win_ratio[i]
            statistics['reward to risk']            = all_reward_to_risk[i]
            statistics['cagr']                      = all_cagr[i]
            statistics['sharpe ratio']              = all_sharpe_ratio[i]
            statistics['sortino ratio']             = all_sortino_ratio[i]
            statistics['mar ratio']                 = all_mar_ratio[i]
            statistics['average return']            = all_average_return[i]
            statistics['stddev returns']            = all_stddev_return[i]
            statistics['average drawdown']          = all_average_dd[i]
            statistics['average drawdown length']   = all_average_dd_length[i]
            statistics['profit factor']             = all_profit_factor[i]
            statistics['pearson correlation']       = all_pearson_correlation[i]
            statistics['max drawdown']              = all_max_drawdown[i]
            statistics['longest drawdown']          = all_longest_dd_period[i]
            statistics['average recovery']          = all_average_recovery[i]
            statistics['longest recovery']          = all_longest_recovery[i]
            statistics['burke ratio']               = all_burke_ratio[i]
            statistics['ulcer index']               = all_ulcer_index[i]
            statistics['martin ratio']              = all_martin_ratio[i]
            all_statistics.append(statistics)
        
        if external_df == False:    
            self.statistics['summary'] = all_statistics
        return all_statistics 
        
    def get_profit_factor(self, input_df = None, external_df = False):
    
        """
        Returns the profit factor for all input data columns.
        """
    
        if 'profit factor' in self.statistics and external_df == False:
            return self.statistics['profit factor']
        
        if external_df == False:
            data = self.data.dropna()     
        else:
            data = input_df  
            
        all_profit_factor = []        
        for i in range(0, len(data.columns)):                       
            df = pd.Series(data.iloc[:,i]).dropna()   
            profit_factor = df[df > 0].sum()/(-df[df < 0].sum()) 
            all_profit_factor.append(profit_factor)
        
        if external_df == False:    
            self.statistics['profit factor'] = all_profit_factor
        return all_profit_factor
        
    def get_reward_to_risk(self, input_df = None, external_df = False):
    
        """
        Returns the average reward to risk ratio for all input columns.   
        """
    
        if 'reward to risk' in self.statistics and external_df == False:
            return self.statistics['reward to risk']
            
        if external_df == False:
            data = self.data.dropna()     
        else:
            data = input_df  
          
        all_reward_to_risk = []        
        for i in range(0, len(data.columns)):                       
            df = pd.Series(data.iloc[:,i]).dropna()     
            reward_to_risk = df[df > 0].mean()/(-df[df < 0].mean()) 
            all_reward_to_risk.append(reward_to_risk)
        
        if external_df == False:        
            self.statistics['reward_to_risk'] = all_reward_to_risk
            
        return all_reward_to_risk
            
    def get_win_ratio(self, input_df = None, external_df = False):
    
        """
        Returns the win ratio (percentage of returns that are positive) for all input columns.   
        """
    
        if 'win ratio' in self.statistics and external_df == False:
            return self.statistics['win ratio']
            
        if external_df == False:
            data = self.data.dropna()     
        else:
            data = input_df  
            
        all_win_ratio = []        
        for i in range(0, len(data.columns)):                       
            df = pd.Series(data.iloc[:,i]).dropna()     
            win_ratio = float(len(df[df > 0]))/float(len(df))
            all_win_ratio.append(win_ratio)
        if external_df == False:    
            self.statistics['win ratio'] = all_win_ratio   
        return all_win_ratio       
 
    def get_rolling_return(self, period, input_df = None, external_df = False):
    
        """
        Calculates the 12 month rolling return for all input columns. 
        Note that you can change this window by changing the ROLLING_PLOT_PERIOD
        literal constant.  
        """
        
        if 'rolling return '+str(period) in self.series and external_df == False:
            return self.series['rolling return '+str(period)]
            
        if external_df == False: 
            data = self.data.dropna().resample('M', how=sum)
        else:
            data = input_df.dropna().resample('M', how=sum)
            
        if external_df == False:
            self.series['rolling return '+str(period)] = pd.rolling_sum(data, int(period)).dropna()
        return pd.rolling_sum(data, int(period)).dropna()
            
    def get_rolling_sharpe_ratio(self, period, input_df = None, external_df = False):
    
        """
        Calculates the 12 month rolling sharpe ratio for all input columns. 
        Note that you can change this window by changing the ROLLING_PLOT_PERIOD
        literal constant.  
        """
        
        if 'rolling sharpe ratio '+str(period) in self.series and external_df == False:
            return self.series['rolling sharpe ratio '+str(period)]
            
        if external_df == False: 
            data = self.data.dropna().resample('M', how=sum)
        else:
            data = input_df.dropna().resample('M', how=sum)
            
        rolling_mean = pd.rolling_mean(data, int(period))
        rolling_std = pd.rolling_std(data, int(period))
        if external_df == False:
            self.series['rolling sharpe ratio '+str(period)] = sqrt(12)*(rolling_mean/rolling_std).dropna()
        return sqrt(12)*(rolling_mean/rolling_std).dropna()
        
    def get_rolling_standard_deviation(self, period, input_df = None, external_df = False):
    
        """
        Calculates the 12 month rolling standard deviation for all input columns. 
        Note that you can change this window by changing the ROLLING_PLOT_PERIOD
        literal constant.  
        """
    
        if 'rolling stddev '+str(period) in self.series and external_df == False:
            return self.series['rolling stddev '+str(period)]
            
        if external_df == False: 
            data = self.data.dropna().resample('M', how=sum)
        else:
            data = input_df.dropna().resample('M', how=sum)
            
        if external_df == False:
            self.series['rolling stddev '+str(period)] = pd.rolling_std(data, int(period)).dropna()
        return pd.rolling_std(data, int(period)).dropna()
                   
    def get_underwater_data(self, input_df = None, external_df = False):
    
        """
        Calculates the underwater plot for all input columns. The underwater
        plot has a value of 0 whenever the time series is outside of a drawdown
        period and goes below zero whenever a drawdown is in progress.  
        """
    
        if 'underwater' in self.series and external_df == False:
            return self.series['underwater']
   
        if external_df == False:
            balance = (1+self.data.dropna()).cumprod()      
        else:
            balance = (1+input_df).cumprod() 
            
        all_underWaterData = []
        names = list(balance.columns.values)
    
        for j in range(0, len(balance.columns)):
            underWaterData = []
            maxBalance = 1.0
            for i in range(0, len(balance.index)):
                if balance.ix[i, j] < maxBalance:
                    drawdown = (maxBalance-balance.ix[i, j])/maxBalance
                else:
                    drawdown = 0
                    maxBalance = balance.ix[i, j]
                underWaterData.append(0-1*drawdown)
            all_underWaterData.append(pd.DataFrame(data=underWaterData, index=balance.index))
        
        result = pd.concat(all_underWaterData, axis=1)
        result.columns = names
        
        if external_df == False:
            self.series['result'] = result
        
        return result
        
    def mean_var_portfolio_optimization(self, covarianceType = SAMPLE_COVARIANCE, minWeight=0,  samples=100, gamma_low=-5, gamma_high=5, plotWeights=False, plotEfficientFrontier=False, saveToFileWeights="", saveToFileFrontier=""):
    
        """
        Makes a mean-variance optimization using the monthly returns from the available 
        data series within the dataframe. Returns a vector containing the weights of the instruments
        to be used. The sum of all the weights is always constrained to 1. Note that the covariance matrix
        used can be the sample covariance, ledoit-wolf shrunk covariance matrix, oas shrunk covariance or 
        shrunk sample covariance.
        """
        
        returns = self.data.dropna()
 
        if covarianceType == SAMPLE_COVARIANCE:
            cov_mat = returns.cov()
        elif covarianceType == LEDOIT_WOLF:
            cov_mat = pd.DataFrame(covariance.ledoit_wolf(returns)[0]) 
        elif covarianceType == OAS:
            cov_mat = pd.DataFrame(covariance.oas(returns)[0]) 
        elif covarainceType == SHRUNK_SAMPLE_COVARIANCE:       
            cov_mat = pd.DataFrame(covariance.shrunk_covariance(returns.cov()))
            
        if covarianceType != SAMPLE_COVARIANCE and covarianceType != LEDOIT_WOLF and covarianceType != OAS and covarianceType != SHRUNK_SAMPLE_COVARIANCE:
            return 0      
    
        n = len(cov_mat)
        Sigma = np.asarray(cov_mat.values)
        w = Variable(len(cov_mat))
        gamma = Parameter(sign='positive')
        mu = returns.mean()
        ret = np.asarray(mu.values).T*w 
        risk = quad_form(w, Sigma)
        prob = Problem(Maximize(ret - gamma*risk), 
               [sum_entries(w) == 1, 
                w >= minWeight])
                
        risk_data = np.zeros(samples)
        ret_data = np.zeros(samples)
        gamma_vals = np.logspace(-5, 5, num=samples)
    
        for i in range(samples):
            gamma.value = gamma_vals[i]
            prob.solve(solver=SCS)
            risk_data[i] = sqrt(risk.value)
            ret_data[i] = ret.value
       
            if i == 0:
                final_w = w.value
                max_sharpe = ret.value/sqrt(risk.value)
            
            if (ret.value/risk.value > max_sharpe):
                final_w = w.value
                max_sharpe = ret.value/sqrt(risk.value)
                           
        weights = []
        for weight in final_w:
            weights.append(float(weight[0]))
            
        if plotWeights == True:
            fig, ax = plt.subplots(figsize=(12,8), dpi=100)
            ax.bar(range(0, len(returns.columns.values)), weights, align="center") 
         
            ax.set_xticks(range(0, len(returns.columns.values)))
            labels_x = list(returns.columns.values)
            ax.set_xticklabels(labels_x)
            
            ax.set_ylabel('Weight assigned (fraction)')
            ax.set_xlabel('Asset name')
            
            if self.use_titles:
                ax.set_title("Min variance portfolio weights")
              
            plt.xticks(rotation=90)
            
            if saveToFileWeights == "":
                plt.show()
            else:
                fig.savefig(saveToFileWeights)
                
        if plotEfficientFrontier == True:
            fig, ax = plt.subplots(figsize=(12,8), dpi=100)
            
            plt.plot(risk_data, ret_data, 'g-')
            for i in range(n):
                plt.plot(sqrt(Sigma[i,i]), np.asarray(mu.values)[i], 'ro')
            plt.xlabel('Standard deviation')
            plt.ylabel('Return')
            plt.show()
                         
            if saveToFileFrontier == "":
                plt.show()
            else:
                fig.savefig(saveToFileFrontier)
    
        return weights

        
    def min_variance_portfolio_optimization(self, covarianceType = SAMPLE_COVARIANCE, minWeight = 0, plotWeights=False, saveToFile=""):
    
        """
        Makes a minimum variance optimization using the monthly returns from the available 
        data series within the dataframe. Returns a vector containing the weights of the instruments
        to be used. The sum of all the weights is always constrained to 1. Note that the covariance matrix
        used can be the sample covariance, ledoit-wolf shrunk covariance matrix, oas shrunk covariance or 
        shrunk sample covariance.
        """
        
        returns = self.data.dropna()
 
        if covarianceType == SAMPLE_COVARIANCE:
            cov_mat = returns.cov()
        elif covarianceType == LEDOIT_WOLF:
            cov_mat = pd.DataFrame(covariance.ledoit_wolf(returns)[0]) 
        elif covarianceType == OAS:
            cov_mat = pd.DataFrame(covariance.oas(returns)[0]) 
        elif covarainceType == SHRUNK_SAMPLE_COVARIANCE:       
            cov_mat = pd.DataFrame(covariance.shrunk_covariance(returns.cov()))
            
        if covarianceType != SAMPLE_COVARIANCE and covarianceType != LEDOIT_WOLF and covarianceType != OAS and covarianceType != SHRUNK_SAMPLE_COVARIANCE:
            return 0      
         
        Sigma = np.asarray(cov_mat.values)
        w = Variable(len(cov_mat))
        risk = quad_form(w, Sigma)
        prob = Problem(Minimize(risk), [sum_entries(w) == 1, w >= minWeight])    
        prob.solve(solver=SCS)
    
        weights = []
        for weight in w.value:
            weights.append(float(weight[0]))
            
        if plotWeights == True:
            fig, ax = plt.subplots(figsize=(12,8), dpi=100)
            ax.bar(range(0, len(returns.columns.values)), weights, align="center") 
         
            ax.set_xticks(range(0, len(returns.columns.values)))
            labels_x = list(returns.columns.values)
            ax.set_xticklabels(labels_x)
            
            ax.set_ylabel('Weight assigned (fraction)')
            ax.set_xlabel('Asset name')
            
            if self.use_titles:
                ax.set_title("Min variance portfolio weights")
              
            plt.xticks(rotation=90)
            
            if saveToFile == "":
                plt.show()
            else:
                fig.savefig(saveToFile)
        
        return weights
    
        
    def plot_monthly_returns_heatmap(self, saveToFile=""):
    
        """
        Plots a heatmap showing the monthly return series. Note that 
        the color scaling is automatic and goes from dark green for the most positive
        to dark red for the most negative. Note that no particular distinction is made
        to make positive values green and/or negative values red. 
        """
        
        returns = self.get_monthly_returns()
        
        for i in range(0, len(returns.columns)):
               
            df = pd.DataFrame(returns.ix[:,i]*100)
            df['month']= df.index.month
            df['year']= df.index.year
            
            heatmap_data = pd.pivot_table(df, index='year', columns='month', values=returns.columns[i])
            labels_y = heatmap_data.index
            labels_x = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']

            fig = plt.figure(figsize=(14,12))              
            ax = plt.subplot(1,1,1)
            
            heatmap = ax.matshow(heatmap_data, aspect = 'auto', origin = 'lower', cmap ="RdYlGn")
            
            ax.set_yticks(np.arange(heatmap_data.shape[0]) + 0.5, minor=False)
            ax.set_xticks(np.arange(heatmap_data.shape[1]) + 0.5, minor=False)         
            
            ax.set_xticklabels(labels_x, minor=False)
            ax.set_yticklabels(labels_y, minor=False)
            
            for y in range(heatmap_data.shape[0]):
                for x in range(heatmap_data.shape[1]):
                    if not np.isnan(heatmap_data.iloc[y, x]):
                        plt.text(x, y, '%.2f' % heatmap_data.iloc[y, x],
                                horizontalalignment='center',
                                verticalalignment='center',
                                size=6.0
                                )
                            
            ax.set_ylabel('Year')
            ax.set_xlabel('Month')
                           
            ax.invert_yaxis()
            ax.xaxis.tick_top()
            
            if self.use_titles:
                plt.text(0.5, 1.08, self.data.columns[i], horizontalalignment='center', fontsize=20, transform = ax.transAxes)        
            
            plt.colorbar(heatmap)
            plt.xticks(rotation=90)
            
            if saveToFile == "":
                plt.show()
            else:
                fig.savefig(saveToFile)
            
    def plot_annual_returns(self, saveToFile=""):
    
        """
        Plots annual returns as well as the mean annual return as
        a horizontal dashed black line.
        """
        
        returns = self.get_annual_returns()
        
        for i in range(0, len(returns.columns)):
               
            df = pd.DataFrame(returns.ix[:,i]*100)
            df['year']= df.index.year
                                                   
            fig, ax = plt.subplots(figsize=(12,8), dpi=100)
               
            ax.bar(df['year'].values, df[df.columns[0]].values, align="center") 
            ax.axhline(df[df.columns[0]].values.mean(), linestyle='dashed', color='black', linewidth=1.5)
            
            labels_x = list(set(df['year']))
            ax.set_xticks(labels_x)
            
            ax.set_ylabel('Year Return (%)')
            ax.set_xlabel('Time')
            if self.use_titles:
                ax.set_title(self.data.columns[i])
              
            plt.xticks(rotation=90)
            
            if saveToFile == "":
                plt.show()
            else:
                fig.savefig(saveToFile)
            
    def plot_drawdown_periods(self, saveToFile=""):
    
        """
        Plots all drawdown periods as a function of time.
        The plot also shows each drawdown period's length.
        """
        
        all_drawdowns = self.get_dd_periods()
             
        for j, drawdowns in enumerate(all_drawdowns):
               
            df = drawdowns                                               
            fig, ax = plt.subplots(figsize=(12,8), dpi=100)
               
            ax.bar(list(df['dd_start']), list(df['dd_depth']*100)) 
            ax.axhline(df['dd_depth'].values.mean(), linestyle='dashed', color='black', linewidth=1.5)
            
            for i in range(0, len(df.index)):
                plt.text(df['dd_start'][i], df['dd_depth'][i]*100, '%d' % int(df['dd_length'][i]),
                            horizontalalignment='center',
                            verticalalignment='center',
                            size=6.0
                            )
            
            ax.set_ylabel('Drawdown depth (%)')
            ax.set_xlabel('Time')    
                  
            if self.use_titles:
                ax.set_title(self.data.columns[j])
              
            if saveToFile == "":
                plt.show()
            else:
                fig.savefig(saveToFile)
            
    def plot_drawdown_distribution(self, saveToFile=""):
    
        """
        Plots a distribution of drawdown period depths.
        The plot always uses 15 bins.
        """
        
        drawdowns = self.get_dd_period_depths()
        
        for i, df in enumerate(drawdowns):
        
            fig, ax = plt.subplots(figsize=(20,14), dpi=100)
               
            n, bins, patches = ax.hist(df*100, bins=15, alpha=0.75) 
            ax.axvline(df.mean(), linestyle='dashed', color="black")
                       
            ax.set_xlabel('Drawdowns (%)')
            ax.set_ylabel('Frequency')
            
            if self.use_titles:
                ax.set_title(self.data.columns[i])
             
            plt.xticks(rotation=90)
            if saveToFile == "":
                plt.show()
            else:
                fig.savefig(saveToFile)
     
    def plot_drawdown_length_distribution(self, saveToFile=""):
    
        """
        Plots a distribution of drawdown period lengths.
        The plot always uses 15 bins.
        """
        
        drawdowns = self.get_dd_period_lengths()
        
        for i, df in enumerate(drawdowns):
        
            fig, ax = plt.subplots(figsize=(20,14), dpi=100)
               
            n, bins, patches = ax.hist(df, bins=15, alpha=0.75) 
            ax.axvline(df.mean(), linestyle='dashed', color="black")
                       
            ax.set_xlabel('Drawdown length (days)')
            ax.set_ylabel('Frequency')
            
            if self.use_titles:
                ax.set_title(self.data.columns[i])
             
            plt.xticks(rotation=90)
            if saveToFile == "":
                plt.show()
            else:
                fig.savefig(saveToFile)
    
    def plot_correlation_heatmap(self, saveToFile=""):
    
        """
        Plots a correlation heatmap showing the correlations
        between the different input columns. 
        """
        
        monthlyReturns = self.get_monthly_returns()
        correlations = monthlyReturns.corr()
                    
        fig = plt.figure(figsize=(14,12))              
        ax = plt.subplot(1,1,1)
            
        heatmap = ax.matshow(correlations, aspect = 'auto', origin = 'lower', cmap ="RdYlGn")
        ax.invert_yaxis()
        ax.xaxis.tick_top()
        
        for y in range(correlations.shape[0]):
            for x in range(correlations.shape[1]):
                plt.text(x, y, '%.2f' % correlations.ix[y, x],
                        horizontalalignment='center',
                        verticalalignment='center',
                        size=6.0
                        )
        
        if saveToFile == "":
            plt.show()
        else:
            fig.savefig(saveToFile)
            
    def get_dd_periods(self, input_df = None, external_df = False):
    
        """
        Gets all drawdown periods as a dictionary containing drawdown
        period start, drawdown period end, drawdown depth and drawdown
        length for all input data columsn. 
        """
    
        if 'drawdown periods' in self.series and external_df == False:
            return self.series['drawdown periods']
            
        if external_df == False:
            balance = (1+self.data.dropna()).cumprod()      
        else:
            balance = (1+input_df).cumprod() 
             
        all_dd_periods = []
        
        for j in range(0, len(balance.columns)):
        
            all_drawdown_lengths = []
            all_drawdown_depths = []
            all_drawdown_start = []
            all_drawdown_end = []
            
            drawdownStart = balance.index[0] 
            maxBalance = 1.0
            drawdown = 0.0
            bottom = 0.0
            
            for i in range(0, len(balance.index)):
                if balance.ix[i, j] < maxBalance:
                    drawdown = (maxBalance-balance.ix[i, j])/maxBalance              
                else:
                    if bottom > 0.0:
                        drawdownEnd = balance.index[i]
                        difference = drawdownEnd-drawdownStart
                        all_drawdown_lengths.append(difference.days)
                        all_drawdown_depths.append(bottom)
                        all_drawdown_start.append(drawdownStart)
                        all_drawdown_end.append(drawdownEnd)
                    drawdown = 0.0                  
                    bottom = 0.0
                    drawdownStart = balance.index[i]              
                    maxBalance = balance.ix[i, j]
                if drawdown > bottom:
                    bottom = drawdown  
            
            if bottom > 0.0:
                drawdownEnd = balance.index[-1]
                difference = drawdownEnd-drawdownStart
                all_drawdown_lengths.append(difference.days)
                all_drawdown_depths.append(bottom)
                all_drawdown_start.append(drawdownStart)
                all_drawdown_end.append(drawdownEnd)                       
                        
            dd_periods_summary = {'dd_start': all_drawdown_start, 'dd_end': all_drawdown_end, 'dd_depth': all_drawdown_depths, 'dd_length': all_drawdown_lengths}   
                                                                        
            all_dd_periods.append(pd.DataFrame.from_dict(dd_periods_summary))
         
        if external_df == False:    
            self.series['drawdown periods'] = all_dd_periods 
                          
        return all_dd_periods      
            
    def plot_monthly_returns(self, saveToFile=""):
    
        """
        Plots monthly returns. The mean monthly return is also
        plotted as a black dashed horizontal line.
        """
        
        returns = self.get_monthly_returns()
        
        for i in range(0, len(returns.columns)):
               
            df = pd.Series(returns.ix[:,i]*100)
                                                   
            fig, ax = plt.subplots(figsize=(20,14), dpi=100)
               
            ax.bar(df.index, df, 30, align="center") 
            ax.axhline(df.values.mean(), linestyle='dashed', color='black', linewidth=1.5)
            
            ax.xaxis.set_major_formatter(DateFormatter('%m-%Y'))
            
            ax.set_ylabel('Month Return (%)')
            ax.set_xlabel('Time')
            
            if self.use_titles:
                ax.set_title(self.data.columns[i])
                           
            plt.xticks(rotation=90)
            if saveToFile == "":
                plt.show()
            else:
                fig.savefig(saveToFile)
            
    def plot_monthly_return_distribution(self, saveToFile=""):
    
        """
        Plots monthly return distribution. The mean monthly return is also
        plotted as a black dashed vertical line. The distribution uses
        15 bins.
        """
        
        returns = self.get_monthly_returns()
        
        for i in range(0, len(returns.columns)):
               
            df = pd.Series(returns.ix[:,i]*100)
                                                   
            fig, ax = plt.subplots(figsize=(20,14), dpi=100)
               
            n, bins, patches = ax.hist(df, bins=15, alpha=0.75, normed=True) 
            ax.axvline(df.mean(), linestyle='dashed', color="black")
            
            y = mlab.normpdf( bins, df.mean(), df.std())
            l = plt.plot(bins, y, 'r--', linewidth=2)
            
            ax.set_xlabel('Month Return (%)')
            ax.set_ylabel('Normalized Frequency')
            
            if self.use_titles:
                ax.set_title(self.data.columns[i])
             
            plt.xticks(rotation=90)
            if saveToFile == "":
                plt.show()
            else:
                fig.savefig(saveToFile)
                 
    def plot_analysis_returns(self, saveToFile=""):
    
        """
        Plots the cumulative return, the underwater plot and
        the weekly returns on a single graph. This plot
        is similar to the PerformanceAnalytics R library summary plot.
        In addition the plot also includes lines to highlight 
        the deepest drawdown periods for all loaded time series.
        """
    
        data = self.data.dropna()
        balance =(1+data).cumprod()
        weeklyReturns = self.get_weekly_returns()     
              
        max_drawdown_start, max_drawdown_end  = self.get_max_dd_dates()

        underWaterSeries = self.get_underwater_data()
        
        fig = plt.figure(figsize=(10,10))
              
        ax1 = plt.subplot2grid((10, 1), (0, 0), rowspan=4)
        ax2 = plt.subplot2grid((10, 1), (5, 0), rowspan=2)
        ax3 = plt.subplot2grid((10, 1), (8, 0), rowspan=2)
        
        ax1.set_yscale('log')
        
        for axis in [ax1.xaxis, ax1.yaxis]:
            axis.set_major_formatter(ScalarFormatter())
        
        ax1.xaxis.set_ticklabels([])
        ax2.xaxis.set_ticklabels([])
        
        color_cycle = ax1._get_lines.prop_cycler
        colors = []
        
        for i, color in enumerate(color_cycle):
            if i<len(max_drawdown_start):
                colors.append(color['color'])
            else:
                break
                
        ax1.set_prop_cycle(None)    
        
        for i, column in enumerate(balance.columns):  
            if self.use_titles:                
                ax1.plot(balance.index, balance[column], label=self.data.columns[i])   
            else:
                ax1.plot(balance.index, balance[column])       
               
        ax2.plot(weeklyReturns.index, weeklyReturns)
        ax3.plot(underWaterSeries.index, underWaterSeries)
        
        if self.use_titles:
            ax1.legend(loc='upper center', bbox_to_anchor=(0.5, 1.3), ncol= 3, fancybox=True, shadow=True)
        
        ax3.set_xlabel('Time')
        ax1.set_ylabel('Cumulative return')
        ax2.set_ylabel('Week return')
        ax3.set_ylabel('Drawdown')
                    
        #self.timeseries['rolling_maxdd'].plot(label='max_dd')
        for i in range(0, len(max_drawdown_start)):
            ax1.axvline(max_drawdown_start[i], linestyle='dashed', color=colors[i])
            ax1.axvline(max_drawdown_end[i], linestyle='dashed', color=colors[i])       
        
        ax1.axhline(1.0, linestyle='dashed', color='black', linewidth=1.5)
        ax2.axhline(0.0, linestyle='dashed', color='black', linewidth=1.5)
        ax3.axhline(0.0, linestyle='dashed', color='black', linewidth=1.5)
              
        ax1.yaxis.grid(b=True, which='major', color='grey', linewidth=0.5, linestyle='dashed')
        ax1.xaxis.grid(b=True, which='major', color='grey', linewidth=0.5, linestyle='dashed')
 
        ax2.yaxis.grid(b=True, which='major', color='grey', linewidth=0.5, linestyle='dashed')
        ax2.xaxis.grid(b=True, which='major', color='grey', linewidth=0.5, linestyle='dashed')

        ax3.yaxis.grid(b=True, which='major', color='grey', linewidth=0.5, linestyle='dashed')
        ax3.xaxis.grid(b=True, which='major', color='grey', linewidth=0.5, linestyle='dashed')
 
        ax1.set_axisbelow(True)  
        
        plt.legend()
        if saveToFile == "":
            plt.show()
        else:
            fig.savefig(saveToFile)
            
    def get_portfolio_returns(self, weights):
    
        """
        Calculates the return dataframe of a portfolio
        made up of all the columns within the dataframe
        using the specified vector of weights.
        """
           
        data = self.data.dropna()
        
        for i in range(0, len(data.columns)): 
            data.ix[:,i] = data.ix[:,i]*weights[i]
            
        data = pd.DataFrame(data.sum(axis=1))
        
        return data
            
    def plot_analysis_portfolio_returns(self, weights, saveToFile=""):
    
        """
        Plots the cumulative return, the underwater plot and
        the weekly returns of a portfolio on a single graph. This plot
        is similar to the PerformanceAnalytics R library summary plot.
        In addition the plot also includes lines to highlight 
        the deepest drawdown periods for all loaded time series.
        The weights are specified through a python list containing
        a separate weight for each DataFrame column.
        """
    
        data = self.get_portfolio_returns(weights)
        balance =(1+data).cumprod()
        
        weeklyReturns = self.get_weekly_returns(input_df=data, external_df=True)     
              
        max_drawdown_start, max_drawdown_end  = self.get_max_dd_dates(input_df=data, external_df=True)

        underWaterSeries = self.get_underwater_data(input_df=data, external_df=True)
        
        fig = plt.figure(figsize=(10,10))
              
        ax1 = plt.subplot2grid((10, 1), (0, 0), rowspan=4)
        ax2 = plt.subplot2grid((10, 1), (5, 0), rowspan=2)
        ax3 = plt.subplot2grid((10, 1), (8, 0), rowspan=2)
        
        ax1.set_yscale('log')
        
        for axis in [ax1.xaxis, ax1.yaxis]:
            axis.set_major_formatter(ScalarFormatter())
        
        ax1.xaxis.set_ticklabels([])
        ax2.xaxis.set_ticklabels([])
        
        color_cycle = ax1._get_lines.prop_cycler
        colors = []
        
        for i, color in enumerate(color_cycle):
            if i<len(max_drawdown_start):
                colors.append(color['color'])
            else:
                break
                
        ax1.set_prop_cycle(None)    
        
        for i, column in enumerate(balance.columns):  
            if self.use_titles:                
                ax1.plot(balance.index, balance[column], label=self.data.columns[i])   
            else:
                ax1.plot(balance.index, balance[column])       
               
        ax2.plot(weeklyReturns.index, weeklyReturns)
        ax3.plot(underWaterSeries.index, underWaterSeries)
             
        ax3.set_xlabel('Time')
        ax1.set_ylabel('Cumulative return')
        ax2.set_ylabel('Week return')
        ax3.set_ylabel('Drawdown')
                    
        #self.timeseries['rolling_maxdd'].plot(label='max_dd')
        for i in range(0, len(max_drawdown_start)):
            ax1.axvline(max_drawdown_start[i], linestyle='dashed', color=colors[i])
            ax1.axvline(max_drawdown_end[i], linestyle='dashed', color=colors[i])       
        
        ax1.axhline(1.0, linestyle='dashed', color='black', linewidth=1.5)
        ax2.axhline(0.0, linestyle='dashed', color='black', linewidth=1.5)
        ax3.axhline(0.0, linestyle='dashed', color='black', linewidth=1.5)
              
        ax1.yaxis.grid(b=True, which='major', color='grey', linewidth=0.5, linestyle='dashed')
        ax1.xaxis.grid(b=True, which='major', color='grey', linewidth=0.5, linestyle='dashed')
 
        ax2.yaxis.grid(b=True, which='major', color='grey', linewidth=0.5, linestyle='dashed')
        ax2.xaxis.grid(b=True, which='major', color='grey', linewidth=0.5, linestyle='dashed')

        ax3.yaxis.grid(b=True, which='major', color='grey', linewidth=0.5, linestyle='dashed')
        ax3.xaxis.grid(b=True, which='major', color='grey', linewidth=0.5, linestyle='dashed')
 
        ax1.set_axisbelow(True)  
        
        plt.legend()
        if saveToFile == "":
            plt.show()
        else:
            fig.savefig(saveToFile)
        
    def plot_analysis_rolling(self, saveToFile=""):
    
        """
        Plots the 12 month rolling return, sharpe ratio and standard deviation.
        This is similar to the rolling performance summary plot available in
        the PerformanceAnalytics R library.
        """
        
        data = self.data.dropna()
        
        rollingAnnualReturn = self.get_rolling_return(ROLLING_PLOT_PERIOD)
        rollingAnnualSharpeRatio = self.get_rolling_sharpe_ratio(ROLLING_PLOT_PERIOD)
        rollingAnnualStandardDeviation = self.get_rolling_standard_deviation(ROLLING_PLOT_PERIOD)
                        
        fig = plt.figure(figsize=(12,10))
        
        ax1 = plt.subplot2grid((12, 1), (0, 0), rowspan=3)
        ax2 = plt.subplot2grid((12, 1), (4, 0), rowspan=3)
        ax3 = plt.subplot2grid((12, 1), (8, 0), rowspan=3)
        
        color_cycle = ax1._get_lines.prop_cycler
        colors = []
        
        for i, color in enumerate(color_cycle):
            if i<len(data.columns):
                colors.append(color['color'])
            else:
                break
                
        ax1.set_prop_cycle(None)
             
        for axis in [ax1.xaxis, ax1.yaxis]:
            axis.set_major_formatter(ScalarFormatter())
        
        ax1.xaxis.set_ticklabels([])
        ax2.xaxis.set_ticklabels([]) 
        
        for i, column in enumerate(rollingAnnualReturn.columns): 
            if self.use_titles:                  
                ax1.plot(rollingAnnualReturn.index, rollingAnnualReturn[column], label=self.data.columns[i]) 
            else:
                ax1.plot(rollingAnnualReturn.index, rollingAnnualReturn[column])   
            
        ax2.plot(rollingAnnualSharpeRatio.index, rollingAnnualSharpeRatio)
        ax3.plot(rollingAnnualStandardDeviation.index, rollingAnnualStandardDeviation)
        
        if self.use_titles: 
            ax1.legend(loc='upper center', bbox_to_anchor=(0.5, 1.3), ncol=3, fancybox=True, shadow=True)
        
        
        ax3.set_xlabel('Time')
        ax1.set_ylabel(str(ROLLING_PLOT_PERIOD)+'M Return')
        ax2.set_ylabel(str(ROLLING_PLOT_PERIOD)+'M Sharpe')
        ax3.set_ylabel(str(ROLLING_PLOT_PERIOD)+'M StdDev')
                    
        ax1.axhline(0.0, linestyle='dashed', color='black', linewidth=1.5)
        ax2.axhline(0.0, linestyle='dashed', color='black', linewidth=1.5)
        ax3.axhline(0.0, linestyle='dashed', color='black', linewidth=1.5)
        
        for i in range(0, len(data.columns)):
            ax1.axhline(rollingAnnualReturn.ix[:,i].mean(), linestyle='dashed', color=colors[i])
            ax2.axhline(rollingAnnualSharpeRatio.ix[:,i].mean(), linestyle='dashed', color=colors[i])
            ax3.axhline(rollingAnnualStandardDeviation.ix[:,i].mean(), linestyle='dashed', color=colors[i])
              
        ax1.yaxis.grid(b=True, which='major', color='grey', linewidth=0.5, linestyle='dashed')
        ax1.xaxis.grid(b=True, which='major', color='grey', linewidth=0.5, linestyle='dashed')
 
        ax2.yaxis.grid(b=True, which='major', color='grey', linewidth=0.5, linestyle='dashed')
        ax2.xaxis.grid(b=True, which='major', color='grey', linewidth=0.5, linestyle='dashed')

        ax3.yaxis.grid(b=True, which='major', color='grey', linewidth=0.5, linestyle='dashed')
        ax3.xaxis.grid(b=True, which='major', color='grey', linewidth=0.5, linestyle='dashed')
 
        ax1.set_axisbelow(True) 
              
        plt.legend()
        if saveToFile == "":
            plt.show()
        else:
            fig.savefig(saveToFile)
        
    def get_log_returns(self, input_df = None, external_df = False):
    
        """
        Returns a dataframe containing the logarithmic returns for all
        input time series.
        """
    
        if 'log returns' in self.series and external_df == False:
            return self.series['log returns']
            
        if external_df == False:
            data = self.data.dropna()     
        else:
            data = input_df  
            
        data = (1+self.data).cumprod()
        data = np.log(data) - np.log(data.shift(1)).dropna()
        
        if external_df == False:
            self.series['log returns'] = data
        
        return data

    def get_returns_avg(self, input_df = None, external_df = False):
    
        """
        Returns the mean return value for all input time series.
        """
    
        if 'average return' in self.statistics and external_df == False:
            return self.statistics['average return']
            
        if external_df == False:
            data = self.data.dropna()     
        else:
            data = input_df  
            
        if external_df == False:
            self.statistics['average return'] = data.mean().values
        return data.mean().values

    def get_returns_std(self, input_df = None, external_df = False):
    
        """
        Returns the standard deviation of returns for all input time series.
        """
    
        if 'stddev returns' in self.statistics and external_df == False:
            return self.statistics['stddev returns']
            
        if external_df == False:
            data = self.data.dropna()     
        else:
            data = input_df  
            
        if external_df == False:
            self.statistics['stddev returns'] = data.std().values
        return data.std().values
        
    def get_sharpe_ratio(self, input_df = None, external_df = False):
    
        """
        Returns the sharpe ratio for all input time series. Note that
        the library assumes that the reference risk free return is zero
        which is common practice.
        """
    
        if 'sharpe ratio' in self.statistics and external_df == False:
            return self.statistics['sharpe ratio']
            
        if external_df == False:
            returns = self.data.dropna()     
        else:
            returns = input_df   
            
        if external_df == False:
            self.statistics['sharpe ratio'] = sqrt(252)*(returns.mean()/returns.std()).values 
              
        return sqrt(252)*(returns.mean()/returns.std()).values   
        
    def get_downside_risk(self, base_return=0.0, input_df=None, external_df = False):
    
        """
        Returns the downside risk for all input time series.
        """
    
        if 'downside risk' in self.series and external_df == False:
            return self.series['downside risk']
                 
        if external_df == False:
            data = self.data.dropna()     
        else:
            data = input_df   
            
        all_risk_diff = []
        names = list(data.columns.values)
        
        for j in range(0, len(data.columns)):
            risk_diff = []
            for i in range(0, len(data.index)):
                if data.ix[i, j] < base_return:
                    risk_diff.append(data.ix[i, j]-base_return)
                else:   
                    risk_diff.append(0.0)
            all_risk_diff.append(pd.DataFrame(data=risk_diff, index=data.index))
        
        result = pd.concat(all_risk_diff, axis=1)
        result.columns = names
        
        if external_df == False:
            self.series['downside risk'] = result
                 
        return result
        
    def get_sortino_ratio(self, base_return=0.0, input_df=None, external_df = False):
    
        """
        Returns the sortino ratio for all input time series.
        """
    
        if 'sortino ratio' in self.statistics and external_df == False:
            return self.statistics['sortino ratio']
            
        if external_df == False:
            data = self.data.dropna()     
        else:
            data = input_df   
        
        downside_risk = self.get_downside_risk(base_return, input_df, external_df)
        
        if external_df == False:
            self.statistics['sortino ratio'] = sqrt(252)*(data.mean()/downside_risk.std()).values
        return self.statistics['sortino ratio']
        
    def get_max_dd_dates(self, input_df = None, external_df = False):
    
        """
        Returns the max drawdown starting and ending dates for all input series.
        """
    
        if ('max drawdown start' in self.statistics and external_df == False) and ('max drawdown end' in self.statistics and external_df == False):
            return self.statistics['max drawdown start'], self.statistics['max drawdown end']    
        
        if external_df == False:
            balance = (1+self.data.dropna()).cumprod()      
        else:
            balance = (1+input_df).cumprod() 
                 
        all_maxdrawdown = []
        all_drawdownStart = []
        all_drawdownEnd = []
    
        for j in range(0, len(balance.columns)):
        
            drawdownData = []
            maxBalance = 1.0
            maxdrawdown = 0.0
            drawdownStart = balance.index[0]
            drawdownEnd = balance.index[0]
            previousHighDate = balance.index[0]
            
            for i in range(0, len(balance.index)):
                if balance.ix[i, j] < maxBalance:
                    drawdown = (maxBalance-balance.ix[i, j])/maxBalance              
                else:
                    drawdown = 0
                    maxBalance = balance.ix[i, j]  
                    previousHighDate = balance.index[i]   
                if drawdown > maxdrawdown:
                    maxdrawdown = drawdown
                    drawdownEnd = balance.index[i]
                    drawdownStart = previousHighDate 
                    
            all_drawdownStart.append(drawdownStart)
            all_drawdownEnd.append(drawdownEnd)
        
        if external_df == False:
            self.statistics['max drawdown start'] = all_drawdownStart
            self.statistics['max drawdown end'] = all_drawdownEnd
        
        return all_drawdownStart, all_drawdownEnd
        
    def get_max_dd(self, input_df = None, external_df = False):
    
        """
        Returns the max drawdown value for all input time series.
        """
    
        if 'max drawdown' in self.statistics and external_df == False:
            return self.statistics['max drawdown']
        if external_df == False:
            self.statistics['max drawdown'] = [np.amax(x) for x in self.get_dd_period_depths(input_df, external_df)]
        return  [np.amax(x) for x in self.get_dd_period_depths(input_df, external_df)]
           
    def get_longest_dd_period(self, input_df = None, external_df = False):
    
        """
        Returns the longest drawdown period value in days for all input time series.
        """
    
        if 'longest drawdown' in self.statistics and external_df == False:
            return self.statistics['longest drawdown']
        if external_df == False:
            self.statistics['longest drawdown'] = [np.amax(x) for x in self.get_dd_period_lengths(input_df, external_df)]
        return [np.amax(x) for x in self.get_dd_period_lengths(input_df, external_df)]
        
    def get_longest_recovery(self, input_df = None, external_df = False):
    
        """
        Returns the longest drawdown period recovery value in days for all input time series.
        """
    
        if 'longest recovery' in self.statistics and external_df == False:
            return self.statistics['longest recovery']
        if external_df == False:
            self.statistics['longest recovery'] = [np.amax(x) for x in self.get_recovery_period_lengths(input_df, external_df)]
        return [np.amax(x) for x in self.get_recovery_period_lengths(input_df, external_df)]
    
    def get_average_dd(self, input_df = None, external_df = False):     
    
        """
        Returns the average drawdown period depth value for all input time series.
        """
    
        if 'average drawdown' in self.statistics and external_df == False:
            return self.statistics['average drawdown']
        if external_df == False:
            self.statistics['average drawdown'] = [np.mean(x) for x in self.get_dd_period_depths(input_df, external_df)]
        return [np.mean(x) for x in self.get_dd_period_depths(input_df, external_df)]
              
    def get_average_dd_length(self, input_df = None, external_df = False):    
    
        """
        Returns the average drawdown period length value in days for all input time series.
        """
    
        if 'average drawdown length' in self.statistics and external_df == False:
            return self.statistics['average drawdown length']   
        if external_df == False:
            self.statistics['average drawdown length'] = [np.mean(x) for x in self.get_dd_period_lengths(input_df, external_df)]
        return [np.mean(x) for x in self.get_dd_period_lengths(input_df, external_df)]
        
    def get_average_recovery(self, input_df = None, external_df = False):
    
        """
        Returns the average drawdown period recovery length in days for all input time series.
        """
    
        if 'average recovery' in self.statistics and external_df == False:
            return self.statistics['average recovery'] 
        if external_df == False:     
            self.statistics['average recovery'] = [np.mean(x) for x in self.get_recovery_period_lengths(input_df, external_df)] 
        return [np.mean(x) for x in self.get_recovery_period_lengths(input_df, external_df)]  
              
    def get_recovery_period_lengths(self, input_df = None, external_df = False):
    
        """
        Returns all recovery period lengths in days for all input time series.
        """
        
        if 'recovery lengths' in self.series and external_df == False:
            return self.series['recovery lengths']
    
        if external_df == False:
            balance = (1+self.data.dropna()).cumprod()      
        else:
            balance = (1+input_df).cumprod() 
            
        all_recovery_periods = []     

        for j in range(0, len(balance.columns)):
        
            all_recoveries = []
            maxBalance = 1.0
            drawdown = 0.0
            bottom = 0.0
            recoveryStart = balance.index[0]
            
            for i in range(0, len(balance.index)):
                if balance.ix[i, j] < maxBalance:
                    drawdown = (maxBalance-balance.ix[i, j])/maxBalance              
                else:
                    if drawdown != 0.0:
                        recoveryEnd = balance.index[i]
                        difference = recoveryEnd-recoveryStart
                        all_recoveries.append(difference.days)
                    drawdown = 0.0
                    bottom = 0.0
                    maxBalance = balance.ix[i, j] 
                if drawdown > bottom:
                    bottom = drawdown
                    recoveryStart = balance.index[i]
                                                                    
            all_recovery_periods.append(np.asarray(all_recoveries))
        
        if external_df == False:
            self.series['recovery lengths'] = all_recovery_periods
            
        return all_recovery_periods
        
    def get_dd_period_depths(self, input_df = None, external_df = False):
    
        """
        Returns all drawdown period depths for all input time series.
        """
    
        if 'drawdown depths' in self.series and external_df == False:
            return self.series['drawdown depths']
    
        if external_df == False:
            balance = (1+self.data.dropna()).cumprod()      
        else:
            balance = (1+input_df).cumprod() 
             
        all_dd_period_depths = []     

        for j in range(0, len(balance.columns)):
        
            all_drawdowns = []
            maxBalance = 1.0
            drawdown = 0.0
            bottom = 0.0
            
            for i in range(0, len(balance.index)):
                if balance.ix[i, j] < maxBalance:
                    drawdown = (maxBalance-balance.ix[i, j])/maxBalance              
                else:
                    if bottom != 0.0:
                        all_drawdowns.append(bottom)
                    drawdown = 0.0
                    bottom = 0.0
                    maxBalance = balance.ix[i, j]
                if drawdown > bottom:
                    bottom = drawdown   
                    
            if bottom != 0.0:
                all_drawdowns.append(bottom)                      
                                      
            all_dd_period_depths.append(np.asarray(all_drawdowns))
            
        if external_df == False:
            self.series['drawdown depths'] = all_dd_period_depths
            
        return all_dd_period_depths
        
    def get_dd_period_lengths(self, input_df = None, external_df = False):
    
        """
        Returns all drawdown period lengths in days for all input time series.
        """
    
        if 'drawdown lengths' in self.series and external_df == False:
            return self.series['drawdown lengths']
    
        if external_df == False:
            balance = (1+self.data.dropna()).cumprod()      
        else:
            balance = (1+input_df).cumprod()
            
        all_dd_period_lengths = []     

        for j in range(0, len(balance.columns)):
        
            all_drawdown_lengths = []
            drawdownStart = balance.index[0] 
            maxBalance = 1.0
            drawdown = 0.0
            bottom = 0.0
            
            for i in range(0, len(balance.index)):
                if balance.ix[i, j] < maxBalance:
                    drawdown = (maxBalance-balance.ix[i, j])/maxBalance              
                else:
                    if bottom > 0.0:
                        drawdownEnd = balance.index[i]
                        difference = drawdownEnd-drawdownStart
                        all_drawdown_lengths.append(difference.days)
                    drawdown = 0.0
                    drawdownStart = balance.index[i]              
                    maxBalance = balance.ix[i, j] 
                if drawdown > bottom:
                    bottom = drawdown                       
                        
            if bottom > 0.0:
                drawdownEnd = balance.index[i]
                difference = drawdownEnd-drawdownStart
                all_drawdown_lengths.append(difference.days)
                                      
            all_dd_period_lengths.append(np.asarray(all_drawdown_lengths))
        
        if external_df == False:
            self.series['drawdown lengths'] = all_dd_period_lengths
        
        return all_dd_period_lengths 
        
    def get_cagr(self, input_df = None, external_df = False):
    
        """
        Returns the Compounded Aannual Growth Rate (CAGR) for all input series .
        http://www.investopedia.com/terms/c/cagr.asp
        """
        
        if 'cagr' in self.statistics and external_df == False:
            return self.statistics['cagr']   
        
        if external_df == False:
            balance = (1+self.data.dropna()).cumprod()      
        else:
            balance = (1+input_df).cumprod()
            
        difference  = balance.index[-1] - balance.index[0]
        difference_in_years = (difference.days + difference.seconds/86400)/365.2425
        cagr = ((balance[-1:].values/balance[:1].values)**(1/difference_in_years))-1
        if external_df == False:   
            self.statistics['cagr'] = cagr[0] 
        return cagr[0]
        
    def get_annual_returns(self, input_df = None, external_df = False):
    
        """
        Returns the annual returns for all the input time series.
        """
    
        if 'annual returns' in self.series and external_df == False:
            return self.series['annual returns']
        
        if external_df == False:
            balance = (1+self.data.dropna()).cumprod()      
        else:
            balance = (1+input_df).cumprod()
        
        all_series = []
        for i in range(0, len(balance.columns)):
            series = pd.Series(balance.iloc[:,i])
            annual_return = series.resample('A', how=lastValue).pct_change(fill_method='pad').dropna()
            all_series.append(annual_return)
        annual_returns = pd.concat(all_series, axis=1)
        if external_df == False:
            self.series['annual returns'] = annual_returns
        return annual_returns 
        
    def get_monthly_returns(self, input_df = None, external_df = False):
    
        """
        Returns the monthly returns for all the input time series.
        """
    
        if 'monthly returns' in self.series and external_df == False:
            return self.series['monthly returns']
       
        if external_df == False:
            balance = (1+self.data.dropna()).cumprod()      
        else:
            balance = (1+input_df).cumprod()
       
        all_series = []
        for i in range(0, len(balance.columns)):
            series = pd.Series(balance.iloc[:,i])
            monthly_return = series.resample('M', how=lastValue).pct_change(fill_method='pad').dropna()
            all_series.append(monthly_return)
        monthly_returns = pd.concat(all_series, axis=1)
        if external_df == False:
            self.series['monthly returns'] = monthly_returns
        return monthly_returns 
        
    def get_weekly_returns(self, input_df = None, external_df = False):
        
        """
        Returns the weekly returns for all the input time series.
        """
        
        if 'weekly returns' in self.series and external_df == False:
            return self.series['weekly returns']
        
        if external_df == False:
            balance = (1+self.data.dropna()).cumprod()      
        else:
            balance = (1+input_df).cumprod()
        
        all_series = []
        for i in range(0, len(balance.columns)):
            series = pd.Series(balance.iloc[:,i])
            weekly_return = series.resample('W', how=lastValue).pct_change(fill_method='pad').dropna()
            all_series.append(weekly_return)
        weekly_returns = pd.concat(all_series, axis=1)
        if external_df == False:
            self.series['weekly returns'] = weekly_returns
        return weekly_returns 
        
    def get_pearson_correlation(self, input_df = None, external_df = False):
    
        """
        Returns the pearson correlation coefficient (R) for all input time series.
        """
    
        if 'pearson correlation' in self.statistics and external_df == False:
            return self.statistics['pearson correlation']   
        
        if external_df == False:
            balance = (1+self.data.dropna()).cumprod()      
        else:
            balance = (1+input_df).cumprod()
        
        balance = np.log(balance)
        all_r_values = []
        for i in range(0, len(balance.columns)):
            series = pd.Series(balance.iloc[:,i])
            x = np.array(series.index.astype(np.int64)/(10**9))
            y = np.array(series.values)
            r_value = np.corrcoef(x, y)[0, 1]
            all_r_values.append(r_value)
        
        if external_df == False:
            self.statistics['pearson correlation'] = all_r_values
        return all_r_values
    
    def get_mar_ratio(self, input_df = None, external_df = False):
    
        """
        Returns the MAR ratio for all the input time series.
        """
        
        if 'mar ratio' in self.statistics and external_df == False:
            return self.statistics['mar ratio']   
        cagr = self.get_cagr(input_df, external_df)
        maximumdrawdown  = self.get_max_dd(input_df, external_df)
        mar_ratio = (cagr/maximumdrawdown) 
        if external_df == False:
            self.statistics['mar_ratio'] = mar_ratio          
        return mar_ratio
        
    def get_burke_ratio(self, input_df = None, external_df = False):
    
        """
        Returns the burke ratio for all the input time series.
        """
    
        if 'burke ratio' in self.statistics and external_df == False:
            return self.statistics['burke ratio']   
        cagr = self.get_cagr(input_df, external_df)
        all_burke_downside_risks = []
        all_series_drawdowns = self.get_dd_period_depths(input_df, external_df)
        
        for all_drawdowns in all_series_drawdowns:
            burke_downside_risk = 0.0
            for drawdown in all_drawdowns:
                burke_downside_risk += drawdown**2
            burke_downside_risk = sqrt(burke_downside_risk)
            all_burke_downside_risks.append(burke_downside_risk)
                
        burke_ratio = (cagr/all_burke_downside_risks)
        if external_df == False:
            self.statistics['burke ratio'] = burke_ratio 
        return burke_ratio
        
    def get_ulcer_index(self, input_df = None, external_df = False):
    
        """
        Returns the Ulcer Index for all the input time series.
        """
        
        if 'ulcer index' in self.statistics and external_df == False:
            return self.statistics['ulcer index']   
            
        if external_df == False:
            balance = (1+self.data.dropna()).cumprod()
        else:
            balance = (1+input_df).cumprod()
            
        all_ulcer_index = []
        
        for i in range(0, len(balance.columns)):
            series = pd.Series(balance.iloc[:,i])
            weekly_balance = series.resample('W', how=lastValue)
            sum_squares = 0.0
            max_value = weekly_balance[0]
            for value in weekly_balance:
                if value > max_value:
                    max_value = value
                else:
                    sum_squares = sum_squares + (100.0*((value/max_value)-1.0))**2
                   
            all_ulcer_index.append(sqrt(sum_squares/float(len(weekly_balance))))
        
        if external_df == False:
            self.statistics['ulcer_index'] = all_ulcer_index
        return all_ulcer_index
        
    def get_angle_coefficient(self, input_df = None, external_df = False):
    
        """
        Calculates the angle coefficient defined here:
        
        for the different input time series.
        """
        
        if external_df == False:
            balance = (1+self.data.dropna()).cumprod()
        else:
            balance = (1+input_df).cumprod()
            
        all_angle_coefficients = []
            
        for i in range(0, len(balance.columns)):
            series = pd.Series(balance.iloc[:,i])
            monthly_balance = series.resample('M', how=lastValue)
            monthly_balance_log = np.log(monthly_balance)
            angles = []
            for j in range(1, len(monthly_balance_log.index)):
                angles.append(degrees(atan(monthly_balance_log[j]-monthly_balance_log[j-1])))
                  
            all_angle_coefficients.append(np.asarray(angles).std())
            
        if external_df == False:
            self.statistics['angle_coefficient'] = all_angle_coefficients
        return all_angle_coefficients
        
    
    def get_martin_ratio(self, input_df = None, external_df = False):
        
        """
        Returns the martin ratio for all the input time series.
        """
        
        if 'martin ratio' in self.statistics and external_df == False:
            return self.statistics['martin ratio']   
        cagr = self.get_cagr(input_df, external_df)
        ulcer_index  = self.get_ulcer_index(input_df, external_df)
        martin_ratio = (cagr/ulcer_index)        
        if external_df == False:
            self.statistics['martin ratio'] = martin_ratio         
        return martin_ratio
        
    def get_mc_simulation(self, index=0, period_length=0):
    
        """
        Returns a dataframe containing the returns of a Monte Carlo simulation of the specified length
        using the return data for the specified input data index. The default period length (0) performs a 
        simulation of the same length as the original data series.
        """
           
        dd_periods = self.get_dd_periods()[index]
        last_drawdown_start = dd_periods['dd_start'].iloc[-1]
        df = self.data[self.data.index < last_drawdown_start].dropna()
        
        simulated_returns = []
        
        if period_length == 0:
            period_length = len(df.index)
                 
        mc_df = df.sample(period_length, replace=True)
        mc_df =  mc_df.set_index(df.index[:period_length])       
        
        return mc_df
        
    def get_mc_statistics(self, index=0, iterations=100, confidence=99, period_length=0):
    
        """
        Returns the CAGR and Sharpe for a Monte Carlo simulation with a defined number of iteration
        at a defined confidence and length interval. 
        """
    
        cagr = []
        sharpe = []
        
        for i in range(0, iterations):
            df = self.get_mc_simulation(index, period_length)
            cagr.append(-self.get_cagr(input_df = df, external_df = True))
            sharpe.append(-self.get_sharpe_ratio(input_df = df, external_df = True))
            
        cagr = np.asarray(cagr)
        sharpe = np.asarray(sharpe)
        
        statistics = {'wc_cagr': -np.percentile(cagr, confidence), 'wc_sharpe': -np.percentile(sharpe, confidence)}
        
        return statistics
        
    def get_mc_statistics_for_current_dd(self, index=0, iterations=100, confidence=99):
    
        """
        Returns the CAGR and sharpe statistics resulting from a Monte Carlo simulations 
        for a given number of iterations at a given confidence for a simulation length
        equal to the selected time series' last drawdown period. The selected time
        series is specified via the index parameter. 
        """
    
        mc_cagr = []
        mc_sharpe = []
        
        dd_periods = self.get_dd_periods()[index]
        last_drawdown_start = dd_periods['dd_start'].iloc[-1]
        difference_in_days = len(pd.bdate_range(last_drawdown_start, self.data.index[-1]))
        
        for i in range(0, iterations):
            df = self.get_mc_simulation(index, difference_in_days)
            mc_cagr.append(-self.get_cagr(input_df = df, external_df = True)[0])
            mc_sharpe.append(-self.get_sharpe_ratio(input_df = df, external_df = True)[0])
            
        wc_cagr = np.asarray(mc_cagr)
        wc_sharpe = np.asarray(mc_sharpe)
        
        df1 = self.data[self.data.index > last_drawdown_start].dropna()
        cagr = self.get_cagr(input_df = df1, external_df = True)[0]
        balance =(1+df1).cumprod() 
        sharpe = self.get_sharpe_ratio(input_df = df1, external_df = True)[0]
                  
        statistics = {'wc_cagr': -np.percentile(wc_cagr, confidence), 'wc_sharpe': -np.percentile(wc_sharpe, confidence), 'cagr':cagr, 'sharpe': sharpe}
        
        return statistics
     
    def plot_mc_simulations(self, index=0, iterations=100, saveToFile=""):
    
        """
        Plots the cumulative return curves of a selected number of Monte Carlo
        simulation iterations.
        """
                               
        fig, ax = plt.subplots(figsize=(12,8), dpi=100)
        ax.set_yscale('log')
        ax.axhline(1.0, linestyle='dashed', color='black', linewidth=1.5)
        ax.set_xlabel('Time')
        ax.set_ylabel('Cum Return') 
        
        for i in range(0, iterations):
            df = self.get_mc_simulation(index)
            balance =(1+df).cumprod()        
            ax.plot(balance.index, balance)
        
        if saveToFile == "":
            plt.show()
        else:
            fig.savefig(saveToFile)
           
    def plot_mc_limits(self, index=0, iterations=100, confidence=99, saveToFile=""):
    
        """
        Plots the system's cumulative return curve plus the upper and lower boundaries for balance
        values obtained from a defined number of Monte Carlo iterations at a defined 
        confidence interval for the last drawdown period.
        """
    
        dd_periods = self.get_dd_periods()[index]
        last_drawdown_start = dd_periods['dd_start'].iloc[-1]
        difference_in_days = len(pd.bdate_range(last_drawdown_start, self.data.index[-1]))
                   
        fig, ax = plt.subplots(figsize=(12,8), dpi=100)
        ax.set_yscale('log')
        ax.axhline(1.0, linestyle='dashed', color='black', linewidth=1.5)
        ax.set_xlabel('Time')
        ax.set_ylabel('Cum Return') 
        
        balance_real = (1+self.data.dropna()).cumprod()
        last_balance = list(balance_real.loc[last_drawdown_start])[0]
        index_limits = pd.bdate_range(last_drawdown_start, self.data.index[-1])                 
                                    
        for i in range(0, iterations): 
                      
            df = self.get_mc_simulation(index, difference_in_days)           
            balance =(1+df).cumprod()*last_balance
            
            if i == 0:
                all_balances = balance
            else:           
                all_balances = pd.concat([all_balances, balance], axis=1)
                    
        all_balances = all_balances.transpose() 
        average_balance = all_balances.mean(axis=0)
        worst_balance = all_balances.quantile(q=confidence/100, axis=0)
        all_balances = -all_balances
        best_balance = -all_balances.quantile(q=confidence/100, axis=0)               
                              
        ax.plot(balance_real.index, balance_real)
        ax.plot(index_limits, best_balance, color="green")
        ax.plot(index_limits, worst_balance, color="red")         
        ax.plot(index_limits, average_balance, color="purple")
        
        if saveToFile == "":
            plt.show()
        else:
            fig.savefig(saveToFile)
            
    def plot_mc_wc_evolution_sharpe(self, index=0, iterations=100, confidence=99, max_period_length=1000, saveToFile=""):
                               
        """
        Plots the evolution of the sharpe ratio calculated using a certain number of Monte Carlo simulation
        iterations at a given confidence interval for period length going from 100 to max_period_length
        in 100 period increments.
        """
        
        fig, ax = plt.subplots(figsize=(12,8), dpi=100)
        ax.set_xlabel('Period length (days)')
        ax.set_ylabel('Worst case sharpe') 
        sharpes = []
        periods = []
        
        for i in range(1, int(max_period_length/100)):
            stats = self.get_mc_statistics(index, iterations, confidence, i*100)
            sharpes.append(stats['wc_sharpe']) 
            periods.append(i*100)   
       
        ax.plot(periods, sharpes)
        
        if saveToFile == "":
            plt.show()
        else:
            fig.savefig(saveToFile)
            
    def plot_mc_wc_evolution_cagr(self, index=0, iterations=100, confidence=99, max_period_length=1000, saveToFile=""):
    
        """
        Plots the evolution of the cagr ratio calculated using a certain number of Monte Carlo simulation
        iterations at a given confidence interval for period length going from 100 to max_period_length
        in 100 period increments.
        """
                               
        fig, ax = plt.subplots(figsize=(12,8), dpi=100)
        ax.axhline(0.0, linestyle='dashed', color='black', linewidth=1.5)
        ax.set_xlabel('Period length (days)')
        ax.set_ylabel('Worst case CAGR') 
        cagrs = []
        periods = []
        
        for i in range(1, int(max_period_length/100)):
            stats = self.get_mc_statistics(index, iterations, confidence, i*100)
            cagrs.append(stats['wc_cagr']) 
            periods.append(i*100)   
        
        ax.plot(periods, cagrs)
        
        if saveToFile == "":
            plt.show()
        else:
            fig.savefig(saveToFile)
        
    def plot_mc_distributions(self, index=0, iterations=100, saveToFile=""):
    
        """
        Plots the distribution of returns of a requested number of Monte Carlo simulation
        iterations. The distribution of the original series is also plotted using a thicker black
        line.
        """
                               
        fig, ax = plt.subplots(figsize=(12,8), dpi=100)
        ax.set_xlabel('Returns (%)')
        ax.set_ylabel('Frequency')
            
        for i in range(0, iterations):                
            df = self.get_mc_simulation(index)*100
            df = df.loc[~(df==0).all(axis=1)]
            distribution = np.histogram(df[df.columns[0]].values, bins=40)
            ax.plot(list(distribution[1][:-1]), list(distribution[0]))
            ax.axvline(df[df.columns[0]].mean(), linestyle='dashed', color="black", linewidth=0.2)
            
        df = self.data.dropna() *100
        df = df.loc[~(df==0).all(axis=1)]
        distribution = np.histogram(df[df.columns[0]].values, bins=40)
        ax.plot(list(distribution[1][:-1]), list(distribution[0]), linewidth=3.0, color="black")

        if saveToFile == "":
            plt.show()
        else:
            fig.savefig(saveToFile)
            

            
