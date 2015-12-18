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

__version__ = "1.412"
ROLLING_PLOT_PERIOD = 12

def lastValue(x):
    try:
        reply = x[-1]
    except:
        reply = None
    return reply

class Analizer:

    def __init__(self, df, column_type='return', titles=None):
        
        if column_type == 'price':    
            all_series = []
            df = pd.DataFrame(df)
            for i in range(0, len(df.columns)):
                series = pd.Series(df.iloc[:,i])
                returns = series.pct_change(fill_method='pad')
                all_series.append(returns)
            self.data = pd.concat(all_series, axis=1)             
        elif column_type == 'return':
            self.data = pd.DataFrame(df)
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
    
        if 'summary' in self.statistics and external_df == False:
            return self.statistics['summary']
    
        all_win_ratio               = self.get_win_ratio(input_df) #
        all_reward_to_risk          = self.get_reward_to_risk(input_df)
        all_cagr                    = self.get_cagr(input_df)
        all_sharpe_ratio            = self.get_sharpe_ratio(input_df)
        all_sortino_ratio           = self.get_sortino_ratio(input_df)
        all_mar_ratio               = self.get_mar_ratio(input_df)
        all_average_return          = self.get_returns_avg(input_df)
        all_stddev_return           = self.get_returns_std(input_df) 
        all_average_dd              = self.get_average_dd(input_df)
        all_profit_factor           = self.get_profit_factor(input_df)
        all_pearson_correlation     = self.get_pearson_correlation(input_df)
        all_max_drawdown            = self.get_max_dd(input_df)
        all_average_dd_length       = self.get_average_dd_length(input_df)
        all_longest_dd_period       = self.get_longest_dd_period(input_df)
        all_average_recovery        = self.get_average_recovery(input_df)
        all_longest_recovery        = self.get_longest_recovery(input_df)
        all_burke_ratio             = self.get_burke_ratio(input_df)
        all_ulcer_index             = self.get_ulcer_index(input_df)
        
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
    
        if 'profit factor' in self.statistics and external_df == False:
            return self.statistics['profit factor']
        
        if external_df == False:
            data = self.data.dropna()     
        else:
            data = input_df  
            
        all_profit_factor = []        
        for i in range(0, len(data.columns)):                       
            df = pd.Series(data.iloc[:,i]).dropna()     
            profit_factor = df[df > 0].sum()/abs(df[df < 0].sum()) 
            all_profit_factor.append(profit_factor)
        
        if external_df == False:    
            self.statistics['profit factor'] = all_profit_factor
        return all_profit_factor
        
    def get_reward_to_risk(self, input_df = None, external_df = False):
    
        if 'reward to risk' in self.statistics and external_df == False:
            return self.statistics['reward to risk']
            
        if external_df == False:
            data = self.data.dropna()     
        else:
            data = input_df  
          
        all_reward_to_risk = []        
        for i in range(0, len(data.columns)):                       
            df = pd.Series(data.iloc[:,i]).dropna()     
            reward_to_risk = df[df > 0].mean()/abs(df[df < 0].mean()) 
            all_reward_to_risk.append(reward_to_risk)
        
        if external_df == False:        
            self.statistics['reward_to_risk'] = all_reward_to_risk
            
        return all_reward_to_risk
            
    def get_win_ratio(self, input_df = None, external_df = False):
    
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
                if balance.iloc[i, j] < maxBalance:
                    drawdown = (maxBalance-balance.iloc[i, j])/maxBalance
                else:
                    drawdown = 0
                    maxBalance = balance.iloc[i, j]
                underWaterData.append(0-1*drawdown)
            all_underWaterData.append(pd.DataFrame(data=underWaterData, index=balance.index))
        
        result = pd.concat(all_underWaterData, axis=1)
        result.columns = names
        
        if external_df == False:
            self.series['result'] = result
        
        return result
        
    def plot_monthly_returns_heatmap(self):
        
        returns = self.get_monthly_returns()
        
        for i in range(0, len(returns.columns)):
               
            df = pd.DataFrame(returns.iloc[:,i]*100)
            df['month']= df.index.month
            df['year']= df.index.year
            
            all_year_returns = []
            
            for month in set(df['month']):
                data = pd.DataFrame(df.loc[df['month'] == month, df.columns[0]].values)
                data.columns = [datetime.date(1900, month, 1).strftime('%B')]
                all_year_returns.append(data)
            
            heatmap_data = pd.concat(all_year_returns, axis=1).dropna()
            
            labels_y = list(set(df['year']))
            labels_x = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
                    
            fig = plt.figure(figsize=(7,6))              
            ax = plt.subplot(1,1,1)
            
            heatmap = ax.matshow(heatmap_data, aspect = 'auto', origin = 'lower', cmap ="RdYlGn")
            
            ax.set_yticks(np.arange(heatmap_data.shape[0]) + 0.5, minor=False)
            ax.set_xticks(np.arange(heatmap_data.shape[1]) + 0.5, minor=False)         
            
            ax.set_xticklabels(labels_x, minor=False)
            ax.set_yticklabels(labels_y, minor=False)
            
            for y in range(heatmap_data.shape[0]):
                for x in range(heatmap_data.shape[1]):
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
            plt.show()
            
    def plot_annual_returns(self):
        
        returns = self.get_annual_returns()
        
        for i in range(0, len(returns.columns)):
               
            df = pd.DataFrame(returns.iloc[:,i]*100)
            df['year']= df.index.year
                                                   
            fig, ax = plt.subplots(figsize=(6,4), dpi=100)
               
            ax.bar(df['year'].values, df[df.columns[0]].values, align="center") 
            ax.axhline(df[df.columns[0]].values.mean(), linestyle='dashed', color='black', linewidth=1.5)
            
            labels_x = list(set(df['year']))
            ax.set_xticks(labels_x)
            
            ax.set_ylabel('Year Return (%)')
            ax.set_xlabel('Time')
            if self.use_titles:
                ax.set_title(self.data.columns[i])
              
            plt.xticks(rotation=90)
            plt.show()
            
    def plot_drawdown_periods(self):
        
        all_drawdowns = self.get_dd_periods()
             
        for j, drawdowns in enumerate(all_drawdowns):
               
            df = drawdowns                                               
            fig, ax = plt.subplots(figsize=(6,4), dpi=100)
               
            ax.bar(list(df['dd_start']), list(df['dd_depth']*100)) 
            ax.axhline(df['dd_depth'].values.mean(), linestyle='dashed', color='black', linewidth=1.5)
            
            for i in range(0, len(df.index)):
                plt.text(df['dd_start'][i], df['dd_depth'][i]*100+2, '%d' % int(df['dd_length'][i]),
                            horizontalalignment='center',
                            verticalalignment='center',
                            size=6.0
                            )
            
            ax.set_ylabel('Drawdown depth (%)')
            ax.set_xlabel('Time')    
                  
            if self.use_titles:
                ax.set_title(self.data.columns[j])
              
            plt.show()
            
    def plot_drawdown_distribution(self):
        
        drawdowns = self.get_dd_period_depths()
        
        for i, df in enumerate(drawdowns):
        
            fig, ax = plt.subplots(figsize=(10,7), dpi=100)
               
            n, bins, patches = ax.hist(df*100, bins=15, alpha=0.75) 
            ax.axvline(df.mean(), linestyle='dashed', color="black")
                       
            ax.set_xlabel('Drawdowns (%)')
            ax.set_ylabel('Frequency')
            
            if self.use_titles:
                ax.set_title(self.data.columns[i])
             
            plt.xticks(rotation=90)
            plt.show()
     
    def plot_drawdown_length_distribution(self):
        
        drawdowns = self.get_dd_period_lengths()
        
        for i, df in enumerate(drawdowns):
        
            fig, ax = plt.subplots(figsize=(10,7), dpi=100)
               
            n, bins, patches = ax.hist(df, bins=15, alpha=0.75) 
            ax.axvline(df.mean(), linestyle='dashed', color="black")
                       
            ax.set_xlabel('Drawdown length (days)')
            ax.set_ylabel('Frequency')
            
            if self.use_titles:
                ax.set_title(self.data.columns[i])
             
            plt.xticks(rotation=90)
            plt.show()   
            
    def get_dd_periods(self, input_df = None, external_df = False):
    
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
                if balance.iloc[i, j] < maxBalance:
                    drawdown = (maxBalance-balance.iloc[i, j])/maxBalance              
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
                    maxBalance = balance.iloc[i, j]
                if drawdown > bottom:
                    bottom = drawdown                         
                        
            dd_periods_summary = {'dd_start': all_drawdown_start, 'dd_end': all_drawdown_end, 'dd_depth': all_drawdown_depths, 'dd_length': all_drawdown_lengths}                    
                                      
            all_dd_periods.append(pd.DataFrame.from_dict(dd_periods_summary))
         
        if external_df == False:    
            self.series['drawdown periods'] = all_dd_periods 
                
        return all_dd_periods      
            
    def plot_monthly_returns(self):
        
        returns = self.get_monthly_returns()
        
        for i in range(0, len(returns.columns)):
               
            df = pd.Series(returns.iloc[:,i]*100)
                                                   
            fig, ax = plt.subplots(figsize=(10,7), dpi=100)
               
            ax.bar(df.index, df, 30, align="center") 
            ax.axhline(df.values.mean(), linestyle='dashed', color='black', linewidth=1.5)
            
            ax.xaxis.set_major_formatter(DateFormatter('%m-%Y'))
            
            ax.set_ylabel('Month Return (%)')
            ax.set_xlabel('Time')
            
            if self.use_titles:
                ax.set_title(self.data.columns[i])
                           
            plt.xticks(rotation=90)
            plt.show()
            
    def plot_monthly_return_distribution(self):
        
        returns = self.get_monthly_returns()
        
        for i in range(0, len(returns.columns)):
               
            df = pd.Series(returns.iloc[:,i]*100)
                                                   
            fig, ax = plt.subplots(figsize=(10,7), dpi=100)
               
            n, bins, patches = ax.hist(df, bins=15, alpha=0.75, normed=True) 
            ax.axvline(df.mean(), linestyle='dashed', color="black")
            
            y = mlab.normpdf( bins, df.mean(), df.std())
            l = plt.plot(bins, y, 'r--', linewidth=2)
            
            ax.set_xlabel('Month Return (%)')
            ax.set_ylabel('Normalized Frequency')
            
            if self.use_titles:
                ax.set_title(self.data.columns[i])
             
            plt.xticks(rotation=90)
            plt.show()
                 
    def plot_analysis_returns(self):
        data = self.data.dropna()
        balance =(1+data).cumprod()
        weeklyReturns = self.get_weekly_returns()     
              
        max_drawdown_start, max_drawdown_end  = self.get_max_dd_dates()

        underWaterSeries = self.get_underwater_data()
              
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
        plt.show()
        
    def plot_analysis_rolling(self):
        data = self.data.dropna()
        
        rollingAnnualReturn = self.get_rolling_return(ROLLING_PLOT_PERIOD)
        rollingAnnualSharpeRatio = self.get_rolling_sharpe_ratio(ROLLING_PLOT_PERIOD)
        rollingAnnualStandardDeviation = self.get_rolling_standard_deviation(ROLLING_PLOT_PERIOD)
                        
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
            ax1.axhline(rollingAnnualReturn.iloc[:,i].mean(), linestyle='dashed', color=colors[i])
            ax2.axhline(rollingAnnualSharpeRatio.iloc[:,i].mean(), linestyle='dashed', color=colors[i])
            ax3.axhline(rollingAnnualStandardDeviation.iloc[:,i].mean(), linestyle='dashed', color=colors[i])
              
        ax1.yaxis.grid(b=True, which='major', color='grey', linewidth=0.5, linestyle='dashed')
        ax1.xaxis.grid(b=True, which='major', color='grey', linewidth=0.5, linestyle='dashed')
 
        ax2.yaxis.grid(b=True, which='major', color='grey', linewidth=0.5, linestyle='dashed')
        ax2.xaxis.grid(b=True, which='major', color='grey', linewidth=0.5, linestyle='dashed')

        ax3.yaxis.grid(b=True, which='major', color='grey', linewidth=0.5, linestyle='dashed')
        ax3.xaxis.grid(b=True, which='major', color='grey', linewidth=0.5, linestyle='dashed')
 
        ax1.set_axisbelow(True) 
              
        plt.legend()
        plt.show()
        
    def get_log_returns(self, input_df = None, external_df = False):
    
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
                if data.iloc[i, j] < base_return:
                    risk_diff.append(data.iloc[i, j]-base_return)
                else:   
                    risk_diff.append(0.0)
            all_risk_diff.append(pd.DataFrame(data=risk_diff, index=data.index))
        
        result = pd.concat(all_risk_diff, axis=1)
        result.columns = names
        
        if external_df == False:
            self.series['downside risk'] = result
                 
        return result
        
    def get_sortino_ratio(self, base_return=0.0, input_df=None, external_df = False):
        if 'sortino ratio' in self.statistics and external_df == False:
            return self.statistics['sortino ratio']
            
        if external_df == False:
            data = self.data.dropna()     
        else:
            data = input_df   
               
        downside_risk = self.get_downside_risk(base_return, input_df)
        if external_df == False:
            self.statistics['sortino ratio'] = sqrt(252)*(data.mean()/downside_risk.std()).values
        return self.statistics['sortino ratio']
        
    def get_max_dd_dates(self, input_df = None, external_df = False):
    
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
                if balance.iloc[i, j] < maxBalance:
                    drawdown = (maxBalance-balance.iloc[i, j])/maxBalance              
                else:
                    drawdown = 0
                    maxBalance = balance.iloc[i, j]  
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
        if 'max drawdown' in self.statistics and external_df == False:
            return self.statistics['max drawdown']
        if external_df == False:
            self.statistics['max drawdown'] = [np.amax(x) for x in self.get_dd_period_depths(input_df)]
        return  [np.amax(x) for x in self.get_dd_period_depths(input_df)]
           
    def get_longest_dd_period(self, input_df = None, external_df = False):
        if 'longest drawdown' in self.statistics and external_df == False:
            return self.statistics['longest drawdown']
        if external_df == False:
            self.statistics['longest drawdown'] = [np.amax(x) for x in self.get_dd_period_lengths(input_df)]
        return [np.amax(x) for x in self.get_dd_period_lengths(input_df)]
        
    def get_longest_recovery(self, input_df = None, external_df = False):
        if 'longest recovery' in self.statistics and external_df == False:
            return self.statistics['longest recovery']
        if external_df == False:
            self.statistics['longest recovery'] = [np.amax(x) for x in self.get_recovery_period_lengths(input_df)]
        return [np.amax(x) for x in self.get_recovery_period_lengths(input_df)]
    
    def get_average_dd(self, input_df = None, external_df = False):     
        if 'average drawdown' in self.statistics and external_df == False:
            return self.statistics['average drawdown']
        if external_df == False:
            self.statistics['average drawdown'] = [np.mean(x) for x in self.get_dd_period_depths(input_df)]
        return [np.mean(x) for x in self.get_dd_period_depths(input_df)]
              
    def get_average_dd_length(self, input_df = None, external_df = False):    
        if 'average drawdown length' in self.statistics and external_df == False:
            return self.statistics['average drawdown length']   
        if external_df == False:
            self.statistics['average drawdown length'] = [np.mean(x) for x in self.get_dd_period_lengths(input_df)]
        return [np.mean(x) for x in self.get_dd_period_lengths(input_df)]
        
    def get_average_recovery(self, input_df = None, external_df = False):
        if 'average recovery' in self.statistics and external_df == False:
            return self.statistics['average recovery'] 
        if external_df == False:     
            self.statistics['average recovery'] = [np.mean(x) for x in self.get_recovery_period_lengths(input_df)] 
        return [np.mean(x) for x in self.get_recovery_period_lengths(input_df)]  
              
    def get_recovery_period_lengths(self, input_df = None, external_df = False):
        
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
                if balance.iloc[i, j] < maxBalance:
                    drawdown = (maxBalance-balance.iloc[i, j])/maxBalance              
                else:
                    if drawdown != 0.0:
                        recoveryEnd = balance.index[i]
                        difference = recoveryEnd-recoveryStart
                        all_recoveries.append(difference.days)
                    drawdown = 0.0
                    bottom = 0.0
                    maxBalance = balance.iloc[i, j] 
                if drawdown > bottom:
                    bottom = drawdown
                    recoveryStart = balance.index[i]
                                                                    
            all_recovery_periods.append(np.asarray(all_recoveries))
        
        if external_df == False:
            self.series['recovery lengths'] = all_recovery_periods
            
        return all_recovery_periods
        
    def get_dd_period_depths(self, input_df = None, external_df = False):
    
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
                if balance.iloc[i, j] < maxBalance:
                    drawdown = (maxBalance-balance.iloc[i, j])/maxBalance              
                else:
                    if bottom != 0.0:
                        all_drawdowns.append(bottom)
                    drawdown = 0.0
                    bottom = 0.0
                    maxBalance = balance.iloc[i, j]
                if drawdown > bottom:
                    bottom = drawdown                       
                                      
            all_dd_period_depths.append(np.asarray(all_drawdowns))
            
        if external_df == False:
            self.series['drawdown depths'] = all_dd_period_depths
            
        return all_dd_period_depths
        
    def get_dd_period_lengths(self, input_df = None, external_df = False):
    
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
                if balance.iloc[i, j] < maxBalance:
                    drawdown = (maxBalance-balance.iloc[i, j])/maxBalance              
                else:
                    if bottom > 0.0:
                        drawdownEnd = balance.index[i]
                        difference = drawdownEnd-drawdownStart
                        all_drawdown_lengths.append(difference.days)
                    drawdown = 0.0
                    drawdownStart = balance.index[i]              
                    maxBalance = balance.iloc[i, j] 
                if drawdown > bottom:
                    bottom = drawdown                       
                                      
            all_dd_period_lengths.append(np.asarray(all_drawdown_lengths))
        
        if external_df == False:
            self.series['drawdown lengths'] = all_dd_period_lengths
        
        return all_dd_period_lengths 
        
    def get_cagr(self, input_df = None, external_df = False):
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
        if 'mar ratio' in self.statistics and external_df == False:
            return self.statistics['mar ratio']   
        cagr = self.get_cagr(input_df)
        maximumdrawdown  = self.get_max_dd(input_df)
        mar_ratio = (cagr/maximumdrawdown) 
        if external_df == False:
            self.statistics['mar_ratio'] = mar_ratio          
        return mar_ratio
        
    def get_burke_ratio(self, input_df = None, external_df = False):
        if 'burke ratio' in self.statistics and external_df == False:
            return self.statistics['burke ratio']   
        cagr = self.get_cagr(input_df)
        all_burke_downside_risks = []
        all_series_drawdowns = self.get_dd_period_depths(input_df)
        
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
    
    def get_martin_ratio(self, input_df = None, external_df = False):
        if 'martin ratio' in self.statistics and external_df == False:
            return self.statistics['martin ratio']   
        cagr = self.get_cagr(input_df)
        ulcer_index  = self.get_ulcer_index(input_df)
        martin_ratio = (cagr/ulcer_index)        
        if external_df == False:
            self.statistics['martin ratio'] = martin_ratio         
        return martin_ratio
        
    def get_mc_simulation(self, index=0):
    
        df = self.data.dropna()   
        distribution = np.histogram(df[df.columns[index]].values, bins=20)
        random_hit = distribution[0]*1000
        acc_prob = [0]
        
        for i in range(0, len(random_hit)):
            acc_prob.append(acc_prob[i]+random_hit[i]) 
        
        simulated_returns = []
        
        for i in range(0, len(df)):
            random_selection = randint(0, np.asarray(distribution[0]).sum()*1000)
            
            for j in range(0, len(acc_prob)):
                if random_selection > acc_prob[j]:
                    chosen_class = j
            
            if chosen_class < len(distribution[1]):
                simulated_returns.append(distribution[1][chosen_class]+((distribution[1][chosen_class+1]-distribution[1][chosen_class])/100)*(randint(0,100))) 
            else:
                simulated_returns.append(distribution[1][chosen_class])    
 
        mc_df = pd.DataFrame(simulated_returns, index=df.index)
        
        return mc_df
        
    def get_mc_statistics(self, index=0, iterations=100, confidence=99):
    
        cagr = []
        max_dd = []
        sharpe = []
        
        for i in range(0, iterations):
            df = self.get_mc_simulation(index)
            cagr.append(-self.get_cagr(input_df = df, external_df = True))
            max_dd.append(self.get_max_dd(input_df = df, external_df = True))
            sharpe.append(-self.get_sharpe_ratio(input_df = df, external_df = True))
            
        cagr = np.asarray(cagr)
        max_dd = np.asarray(max_dd)
        sharpe = np.asarray(sharpe)
        
        statistics = {'cagr': -np.percentile(cagr, confidence), 'max_dd': np.percentile(max_dd, confidence), 'sharpe': -np.percentile(sharpe, confidence)}
        
        return statistics
        
    def plot_mc_simulations(self, index=0, iterations=100):
                               
        fig, ax = plt.subplots(figsize=(6,4), dpi=100)
        ax.set_yscale('log')
        ax.axhline(1.0, linestyle='dashed', color='black', linewidth=1.5)
        ax.set_xlabel('Time')
        ax.set_ylabel('Cum Return') 
        
        for i in range(0, iterations):
            df = self.get_mc_simulation(index)
            balance =(1+df).cumprod()        
            ax.plot(balance.index, balance)
        
        plt.show()
        
    def plot_mc_distributions(self, index=0, iterations=100):
                               
        fig, ax = plt.subplots(figsize=(6,4), dpi=100)
        ax.set_xlabel('Returns (%)')
        ax.set_ylabel('Frequency')
        
        for i in range(0, iterations):                
            df = self.get_mc_simulation(index)*100
            distribution = np.histogram(df[df.columns[0]].values, bins=20)
            ax.plot(list(distribution[1][:-1]), list(distribution[0]))
            ax.axvline(df[df.columns[0]].mean(), linestyle='dashed', color="black", linewidth=0.2)

        plt.show()
        
           
    
        
            
            
