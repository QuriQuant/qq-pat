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

__version__ = "1.0"
ROLLING_PLOT_PERIOD = 12

def lastValue(x):
        try:
            reply = x[-1]
        except:
            reply = None
        return reply

class Analizer:

    def __init__(self, df, column_type='return'):
        
        if column_type == 'price':    
            all_series = []
            df = pd.DataFrame(df)
            for i in range(0, len(df.columns)):
                series = pd.Series(df.iloc[:,i])
                returns = series.pct_change(fill_method='pad').dropna()
                all_series.append(returns)
            self.data = pd.concat(all_series, axis=1)             
        elif column_type == 'return':
            self.data = pd.DataFrame(df)
        else:
            raise ValueError('column_type \'{}\' not valid'.format(column_type))

        self.ratios = {}
        self.timeseries = {}
                
    def get_rolling_return(self, period):
        data = self.data.dropna().resample('M', how=sum)
        return pd.rolling_sum(data, int(period)).dropna()
            
    def get_rolling_sharpe_ratio(self, period):
        data = self.data.dropna().resample('M', how=sum)
        rolling_mean = pd.rolling_mean(data, int(period))
        rolling_std = pd.rolling_std(data, int(period))
        
        return sqrt(12)*(rolling_mean/rolling_std).dropna()
        
    def get_rolling_standard_deviation(self, period):
        data = self.data.dropna().resample('M', how=sum)
        return pd.rolling_std(data, int(period)).dropna()
                   
    def get_underwater_data(self):
    
        data = self.data.dropna()
        balance =(1+data).cumprod()
        all_underWaterData = []
        names = list(data.columns.values)
    
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
                                  
            fig, ax = plt.subplots(figsize=(6,4), dpi=100)
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
              
            plt.xticks(rotation=90)
            plt.show()
            
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
             
            plt.xticks(rotation=90)
            plt.show()
                 
    def plot_analysis_returns(self):
        data = self.data.dropna()
        balance =(1+data).cumprod()
        weeklyReturns = self.get_weekly_returns()     
        
        self.ratios['returns_avg'] = self.get_returns_avg()
        self.ratios['max_dd_start'], self.ratios['max_dd_end'], self.ratios['max_dd'] = self.get_max_dd()

        underWaterSeries = self.get_underwater_data()
              
        ax1 = plt.subplot2grid((10, 1), (0, 0), rowspan=4)
        ax2 = plt.subplot2grid((10, 1), (5, 0), rowspan=2)
        ax3 = plt.subplot2grid((10, 1), (8, 0), rowspan=2)
        
        ax1.set_yscale('log')
        
        for axis in [ax1.xaxis, ax1.yaxis]:
            axis.set_major_formatter(ScalarFormatter())
        
        ax1.xaxis.set_ticklabels([])
        ax2.xaxis.set_ticklabels([])
        
        color_cycle = ax1._get_lines.color_cycle
        colors = []
        
        for i, color in enumerate(color_cycle):
            if i<len(self.ratios['max_dd_start']):
                colors.append(color)
            else:
                break
                
        ax1.set_color_cycle(None)
                
        ax1.plot(balance.index, balance)      
        ax2.plot(weeklyReturns.index, weeklyReturns)
        ax3.plot(underWaterSeries.index, underWaterSeries)
        
        ax3.set_xlabel('Time')
        ax1.set_ylabel('Cumulative return')
        ax2.set_ylabel('Week return')
        ax3.set_ylabel('Drawdown')
                    
        #self.timeseries['rolling_maxdd'].plot(label='max_dd')
        for i in range(0, len(self.ratios['max_dd_start'])):
            ax1.axvline(self.ratios['max_dd_start'][i], linestyle='dashed', color=colors[i])
            ax1.axvline(self.ratios['max_dd_end'][i], linestyle='dashed', color=colors[i])       
        
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
        
        color_cycle = ax1._get_lines.color_cycle
        colors = []
        
        for i, color in enumerate(color_cycle):
            if i<len(data.columns):
                colors.append(color)
            else:
                break
                
        ax1.set_color_cycle(None)
             
        for axis in [ax1.xaxis, ax1.yaxis]:
            axis.set_major_formatter(ScalarFormatter())
        
        ax1.xaxis.set_ticklabels([])
        ax2.xaxis.set_ticklabels([])    
        
        ax1.plot(rollingAnnualReturn.index, rollingAnnualReturn)     
        ax2.plot(rollingAnnualSharpeRatio.index, rollingAnnualSharpeRatio)
        ax3.plot(rollingAnnualStandardDeviation.index, rollingAnnualStandardDeviation)
        
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
        
    def get_log_returns(self):
        data = self.data.dropna()
        data = (1+self.data).cumprod()
        data = np.log(data) - np.log(data.shift(1))
        return data.dropna()

    def get_returns_avg(self):
        return self.data.mean()

    def get_returns_std(self):
        return self.data.std()
        
    def get_sharpe_ratio(self):
        returns = self.data.dropna()    
        return sqrt(252)*(returns.mean()/returns.std()).values
        
    def get_downside_risk(self, base_return=0.0):
        data = self.data.dropna()    
        
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
                 
        return result
        
    def get_sortino_ratio(self, base_return=0.0):
        data = self.data.dropna()    
        downside_risk = self.get_downside_risk(base_return)
        return sqrt(252)*(data.mean()/downside_risk.std()).values
        
    def get_max_dd(self):
    
        balance = (1+self.data.dropna()).cumprod()      
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

            all_maxdrawdown.append(maxdrawdown)
            all_drawdownStart.append(drawdownStart)
            all_drawdownEnd.append(drawdownEnd)
        
        return all_drawdownStart, all_drawdownEnd, all_maxdrawdown
        
    def get_average_dd(self):
    
        balance = (1+self.data.dropna()).cumprod() 
        all_average_drawdowns = []     

        for j in range(0, len(balance.columns)):
        
            all_drawdowns = []
            maxBalance = 1.0
            drawdown = 0.0
            drawdownbottom = 0.0
            
            for i in range(0, len(balance.index)):
                if balance.iloc[i, j] < maxBalance:
                    drawdown = (maxBalance-balance.iloc[i, j])/maxBalance              
                else:
                    if drawdownbottom != 0.0:
                        all_drawdowns.append(drawdownbottom)
                    drawdownbottom = 0.0
                    drawdown = 0.0
                    maxBalance = balance.iloc[i, j]                       
                if drawdown > drawdownbottom:
                    drawdownbottom = drawdown
                                      
            all_average_drawdowns.append(np.asarray(all_drawdowns).mean())
        
        return all_average_drawdowns
        
    def get_cagr(self):
        balance = (1+self.data.dropna()).cumprod()     
        difference  = balance.index[-1] - balance.index[0]
        difference_in_years = (difference.days + difference.seconds/86400)/365.2425
        cagr = ((balance[-1:].values/balance[:1].values)**(1/difference_in_years))-1
        return cagr[0]
        
    def get_annual_returns(self):
        balance = (1+self.data.dropna()).cumprod()
        all_series = []
        for i in range(0, len(balance.columns)):
            series = pd.Series(balance.iloc[:,i])
            annual_return = series.resample('A', how=lastValue).pct_change(fill_method='pad').dropna()
            all_series.append(annual_return)
        annual_returns = pd.concat(all_series, axis=1)
        return annual_returns 
        
    def get_monthly_returns(self):
        balance = (1+self.data.dropna()).cumprod()
        all_series = []
        for i in range(0, len(balance.columns)):
            series = pd.Series(balance.iloc[:,i])
            monthly_return = series.resample('M', how=lastValue).pct_change(fill_method='pad').dropna()
            all_series.append(monthly_return)
        monthly_returns = pd.concat(all_series, axis=1)
        return monthly_returns 
        
    def get_weekly_returns(self):
        balance = (1+self.data.dropna()).cumprod()
        all_series = []
        for i in range(0, len(balance.columns)):
            series = pd.Series(balance.iloc[:,i])
            monthly_return = series.resample('W', how=lastValue).pct_change(fill_method='pad').dropna()
            all_series.append(monthly_return)
        monthly_returns = pd.concat(all_series, axis=1)
        return monthly_returns 
        
    def get_pearson_correlation(self):
        balance = (1+self.data.dropna()).cumprod()
        balance = np.log(balance)
        all_r_values = []
        for i in range(0, len(balance.columns)):
            series = pd.Series(balance.iloc[:,i])
            x = np.array(series.index.astype(np.int64)/(10**9))
            y = np.array(series.values)
            r_value = np.corrcoef(x, y)[0, 1]
            all_r_values.append(r_value)
            
        return all_r_values
    
    def get_mar_ratio(self):
        cagr = self.get_cagr()
        _, _, maximumdrawdown  = self.get_max_dd()
        mar_ratio = (cagr/maximumdrawdown)
        return mar_ratio
        
            
            
