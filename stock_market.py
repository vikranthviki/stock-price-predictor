#!/usr/bin/env python
# coding: utf-8

# In[6]:


import numpy as np
import pandas as pd 
import plotly.graph_objects as go
from datetime import datetime
from statistics import mean
import ta
import plotly.express as px
import plotly.subplots as splt
import time
from sklearn.preprocessing import StandardScaler
import datetime
from itertools import product
from statsmodels.tsa.statespace.sarimax import SARIMAX


# In[ ]:





# In[7]:


d2=int(time.mktime((datetime.date.today()).timetuple()))
d1=int(time.mktime((datetime.date.today()-datetime.timedelta(days=360)).timetuple()))
qst=f'https://query1.finance.yahoo.com/v7/finance/download/TSLA?period1={d1}&period2={d2}&interval=1d&events=history&includeAdjustedClose=true'
df=pd.read_csv(qst)
#df=yf.download(tickers='msft',period='4y',interval='1d')
#df=df.reset_index()
df.columns=['Date','Open','High', 'Low' ,'Close','Adj Close','Volume']
df.head(-5)


# In[8]:


def cand_plot(da,open,high,close,low,com):
         fig = go.Figure(data=[go.Candlestick(x=da,open=open,high=high,low=low,close=close)])
         fig.update_layout(title_text=com+' candel chart', font_color="#115382",title_font_family="Times New Roman",xaxis_title="Date",
                                yaxis_title="Stock Price")
 
         return fig
# 

# In[6]:


def get_macd(price, slow, fast, smooth):
    exp1 = price.ewm(span = fast, adjust = False).mean()
    exp2 = price.ewm(span = slow, adjust = False).mean()
    macd = pd.DataFrame(exp1 - exp2).rename(columns = {'Close':'macd'})
    signal = pd.DataFrame(macd.ewm(span = smooth, adjust = False).mean()).rename(columns = {'macd':'signal'})
    hist = pd.DataFrame(macd['macd'] - signal['signal']).rename(columns = {0:'hist'})
    strat=[]
    trend=[]   
    for i in range(1,len(price)):
        if float(hist.iloc[i-1])>0 and float(hist.iloc[i])<0 :
            strat.append('Sell')
        elif  float(hist.iloc[i-1])<0 and float(hist.iloc[i])>0:
            strat.append('Buy')
        else:
            strat.append('Hold')
        if float(macd.iloc[i]) and float(signal.iloc[i])<=0:
            trend.append('Bearish')
        elif float(macd.iloc[i]) and float(signal.iloc[i])>0:
            trend.append('Bullish')
        else:
            trend.append('Neutral')

    d=pd.DataFrame(np.column_stack([strat,trend]), 
                               columns=['MACD_signal','Trend'])
    frames =  [macd, signal, hist,d]
    MACD= pd.concat(frames, join = 'inner', axis = 1)    
    return MACD
# ma=get_macd(df['Close'], 26,12,9)
# len(ma['MACD_signal'])


# In[7]:


def plot_macd(da,prices, macd, signal, hist):
    fig=go.Figure()
    fig_sub = splt.make_subplots(rows=2, cols=1,row_heights=[2,1],subplot_titles=("closing", "MACD"))
    
    fig_sub.add_trace(go.Scatter(x=da, y=prices,mode='lines',name='close',marker = {'color' : 'blue'}),row=1, col=1)
    fig_sub.add_trace(go.Scatter(x=da, y=macd,mode='lines',name='MACD',marker = {'color' : 'orange'}),row=2, col=1)
    fig_sub.add_trace(go.Scatter(x=da, y=signal,mode='lines',name='Signal',marker = {'color' : 'green'}),row=2, col=1)
    fig_sub.add_trace(go.Bar(x=da, y=hist,visible=True,marker = {'color' : 'purple'}),row=2, col=1)
    return fig    
# plot_macd(df['Date'],df['Close'],get_macd(df['Close'], 26,12,9)['macd'], get_macd(df['Close'], 26,12,9)['signal'],
#           get_macd(df['Close'], 26,12,9)['hist'])


# In[8]:


def Heinki_Ashi(op,hi,lo,clo):
    HAc=[]
    HAo=[]
    HAh=[]
    HAl=[]
    HAo.append(op[0]),HAc.append(clo[0]),HAl.append(lo[0]),HAh.append(hi[0])
    for i in range(1,len(op)):
        HAc.append((op[i]+clo[i]+hi[i]+lo[i])/4)
        HAo.append((HAo[i-1]+HAc[i-1])/2)
        HAh.append(max(HAo[i],HAc[i],hi[i]))
        HAl.append(min(HAo[i],HAc[i],lo[i]))
    HA=pd.DataFrame(np.column_stack([HAo,HAh,HAl,HAc]), 
                               columns=['Open', 'High', 'Low','Close'])
   
    return HA
# HA=Heinki_Ashi(df['Open'],df['High'],df['Low'],df['Close'])


# In[9]:


# HA_MACD=get_macd(HA['Close'], 26,12,9)


# In[10]:


def stocastic(close,w,s1,s2):
    st=ta.momentum.StochRSIIndicator(close=close,window=w,smooth1=s1,smooth2=s2,fillna=True)
    s=pd.concat([st.stochrsi(),st.stochrsi_d(),st.stochrsi_k()],axis=1, join='inner')
    return s 
# st_par=stocastic(df['Close'],14,3,3)  
# st_par


# In[11]:


#df1=sto(df['Close'],14,3,3)
# def plot_stoch(da,price, k, d):
#     fig=go.Figure()
#     fig_sub = splt.make_subplots(rows=2, cols=1,row_heights=[2,1],subplot_titles=("Heinki ashi chart", "Stochastic ocsilator"))
#     fig_sub.add_trace(go.Candlestick(x=da,open=HA['Open'],high=HA['High'],low=HA['Low'],close=HA['Close'],name='Heniki ashi'),row=1, col=1)
#     fig_sub.add_trace(go.Scatter(x=da, y=k,mode='lines',name='stoch_k',marker = {'color' : 'orange'}),row=2, col=1)
#     fig_sub.add_trace(go.Scatter(x=da, y=d,mode='lines',name='stoch_d',marker = {'color' : 'green'}),row=2, col=1) 
#     fig_sub.show()
    
# plot_stoch(df['Date'],df['Close'],st_par['stochrsi_k'],st_par['stochrsi_d'])
    


# In[12]:


def HA_strategy(op,hi,cl,lo,HA):
    #ha=Heinki_Ashi(op,hi,cl,lo)
    s=ta.trend.SMAIndicator(close=cl, window=20, fillna= True)
    SMA=s.sma_indicator()
    sc1=stocastic(cl,14,3,3)  
    sc2=stocastic(cl,50,3,3)  
    sig=[]
    #https://www.moneycontrol.com/news/business/markets/technical-classroom-how-to-use-heikin-ashi-candlestick-for-trading-4124291.html
    for i in range(len(op)):
        if HA['Low'].iloc[i]>=HA['Open'].iloc[i] and HA['Open'].iloc[i]<HA['Close'].iloc[i]   and sc1['stochrsi'].iloc[i]>0.5 and sc1['stochrsi'].iloc[i]<0.9 and sc2['stochrsi'].iloc[i]>0.5 and sc2['stochrsi'].iloc[i]<0.9 and SMA[i]<HA['Open'].iloc[i]: 
            sig.append('Buy')
        elif HA['High'].iloc[i]>=HA['Open'].iloc[i] and HA['Open'].iloc[i]>HA['Close'].iloc[i] and SMA[i]>HA['Open'].iloc[i] :
            sig.append('Sell')
        else:
            sig.append('Hold')
            
    return pd.DataFrame(sig,columns=['HA_signal'] )    
# ha=HA_strategy(df['Open'],df['High'],df['Close'],df['Low'])
# ha['HA_signal'].value_counts()


# In[13]:


def bollinger_band(hi,lo,cl,v):
    
    bb=ta.volatility.BollingerBands(close=cl,window= 40, window_dev= 2, fillna=True)
    hband=bb.bollinger_hband() 
    hbandind=bb.bollinger_hband_indicator()
    lbandind=bb.bollinger_lband_indicator() 
    lband=bb.bollinger_lband() 
    volume_idicator=ta.volume.AccDistIndexIndicator(high=hi, low=lo, close=cl, volume=v, fillna=True)
    adi=volume_idicator.acc_dist_index()
    OBV=ta.volume.OnBalanceVolumeIndicator(close=cl, volume=v, fillna=True)
    obv=OBV.on_balance_volume() 
    s20=ta.trend.SMAIndicator(close=cl, window=20, fillna= True)
    s65=ta.trend.SMAIndicator(close=cl, window=65, fillna= True)
    O20=ta.trend.SMAIndicator(close=obv, window=20, fillna= True)
    O65=ta.trend.SMAIndicator(close=obv, window=65, fillna= True)
    A20=ta.trend.SMAIndicator(close=adi, window=20, fillna= True)
    A65=ta.trend.SMAIndicator(close=adi, window=65, fillna= True)
    SMA20=s20.sma_indicator()
    SMA65=s65.sma_indicator()
    OBV20=O20.sma_indicator()
    OBV65=O65.sma_indicator()    
    ADI20=A20.sma_indicator()
    ADI65=A65.sma_indicator()    
    bollinger_signal=[]
    for i in range(len(lband)-1):
        if  SMA65.iloc[i]<cl.iloc[i] and SMA20.iloc[i]<cl.iloc[i] and OBV65.iloc[i]<=obv.iloc[i] and OBV20.iloc[i]<=obv.iloc[i] and ADI65.iloc[i]<=adi.iloc[i] and ADI20.iloc[i]<=adi.iloc[i] :
               bollinger_signal.append('Buy')
        elif SMA65.iloc[i]>cl.iloc[i] and SMA20.iloc[i]>cl.iloc[i] :
               bollinger_signal.append('Sell')
        else:
               bollinger_signal.append('Hold')
                         
    da=df['Date']        
#     fig=go.Figure()
#     fig_sub = splt.make_subplots(rows=3, cols=1,row_heights=[5,3,3],column_widths=[5],subplot_titles=("chart", "ADI","Volume"))
#     fig_sub.add_trace(go.Candlestick(x=da,open=df['Open'],high=hi,low=lo,close=cl,name='candel stick'),row=1, col=1)
#     fig_sub.update_layout(xaxis_rangeslider_visible=False)
#     fig_sub.add_trace(go.Scatter(x=da, y=hband,mode='lines',name='high',marker = {'color' : 'green'}),row=1, col=1)
#     fig_sub.add_trace(go.Scatter(x=da, y=lband,mode='lines',name='low',marker = {'color' : 'red'}),row=1, col=1)
#     fig_sub.add_trace(go.Scatter(x=da, y=obv,mode='lines',name='obv',marker = {'color' : 'yellow'}),row=2, col=1)
#     fig_sub.add_trace(go.Scatter(x=da, y=adi,mode='lines',name='ADI',marker = {'color' : 'blue'}),row=2, col=1)
#     fig_sub.add_trace(go.Bar(x=da, y=v,name='Volume',visible=True,marker = {'color' : 'purple'}),row=3, col=1)
#     fig_sub.update_layout(height=900, width=1000,
#                   title_text="VOLUME STRATEGY")
#     fig_sub.show()        
    bollinger_signal=pd.DataFrame(bollinger_signal)
    bb=pd.concat([hband,lband,hbandind,lbandind,adi,bollinger_signal],axis=1,join='inner')
    bb.columns=['hband','lband','bbihband','bbilband','ADI','BB_signal']
    return bb


# In[14]:


# bb=bollinger_band(df['High'],df['Low'],df['Close'],df['Volume'])
# len(bb['BB_signal'])



# In[15]:


def VWAP(hi,lo,cl,v,Trend):
    vp=ta.volume.VolumeWeightedAveragePrice(high=hi,low=lo,close=cl,volume=v,window=14,fillna=True)
    vwap=vp.volume_weighted_average_price() 
    signal=[]
#     print(len(Trend))
#     print(len(lo))
    for i in range(len(Trend)):
        if vwap.iloc[i]<cl.iloc[i] and Trend.iloc[i]=='Bullish':
            signal.append('Buy')
        elif vwap.iloc[i]>cl.iloc[i] and Trend.iloc[i]=='Bullish' :
            signal.append('Sell')
        #elif vwap.iloc[i]>df['Close'].iloc[i] and Trend.iloc[i]=='Bearish':
            #signal.append('Buy')
        #elif vwap.iloc[i]<df['Close'].iloc[i] and Trend.iloc[i]=='Bearish':   
            #signal.append('Sell')
        else :
            signal.append('Hold')
        
    return pd.DataFrame(signal,columns=['VWAP_signal'] )        
# vwap=VWAP(df['High'],df['Low'],df['Close'],df['Volume'],ma['Trend'])
# vwap.value_counts()


# In[15]:





# In[15]:





# In[16]:


# df_final=pd.concat([df,ha['HA_signal'],ma['MACD_signal'],bb['BB_signal']],axis=1, join='inner')


# In[17]:


# df_final.info()


# In[18]:


#ticker.institutional_holders['% Out'].sum()
#sp500=read.DataReader('^GSPC','yahoo',datetime.date.today()-datetime.timedelta(days=365),datetime.date.today()-datetime.timedelta(days=1))
#df_fundemental=pd.DataFrame(columns=['Tickers','Market_Cap','Industry','Revenue_Groeth'])


# In[15]:




# In[44]:


def forecast(dat,period):
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    from statsmodels.tsa.stattools import adfuller
    from sklearn.preprocessing import StandardScaler
    import numpy as np
    from itertools import product
    from tqdm import tqdm
    from tqdm import tqdm_notebook, tnrange
    def optimize_SARIMA(parameters_list, d, D, s, exog):
        """
            Return dataframe with parameters, corresponding AIC and SSE
        
            parameters_list - list with (p, q, P, Q) tuples
            d - integration order
            D - seasonal integration order
            s - length of season
            exog - the exogenous variable
        """
    
        results = []
    
        for param in tqdm_notebook(parameters_list):
            try: 
                model = SARIMAX(exog, order=(param[0], d, param[1]), seasonal_order=(param[2], D, param[3], s)).fit()
            except:
                continue
            
            aic = model.aic
            results.append([param, aic])
        
        result_df = pd.DataFrame(results)
        result_df.columns = ['(p,q)x(P,Q)', 'AIC']
    #Sort in ascending order, lower AIC is better
        result_df = result_df.sort_values(by='AIC', ascending=True).reset_index(drop=True)
    
        return result_df
    result = adfuller(dat)
    print('ADF Test Statistic: %.2f' % result[0])
    print('5%% Critical Value: %.2f' % result[4]['5%'])
    print('p-value: %.2f' % result[1])
    diff= dat.diff()
    diff.drop([0], axis=0, inplace=True)
    result = adfuller(diff)
    print('ADF Test Statistic: %.2f' % result[0])
    print('5%% Critical Value: %.2f' % result[4]['5%'])
    print('p-value: %.2f' % result[1])
    diff.sort_index(inplace=True),
    #s_d=seasonal_decompose(diff,model='additive',period =int(len(diff)/2))
    #s_d.plot()
    #plt.show()
    #train = diff[:len(df)-period]
    #test = diff[len(df)-period:]
    scaler = StandardScaler()
    train=np.array(diff)
    train = scaler.fit_transform(train.reshape(-1, 1))
   
    p = range(0, 3, 1)
    d = 1
    q = range(0, 3, 1)
    P = range(0, 3, 1)
    D = 1
    Q = range(0, 3, 1)
    s = 4
    parameters = product(p, q, P, Q)
    parameters_list = list(parameters)
    result_df = optimize_SARIMA(parameters_list, 1, 1,4,train)
    k=result_df
    if k['AIC'][1]-k['AIC'][0]>3*(k['AIC'][2]-k['AIC'][1]):
        f_aic=1
    else:
        f_aic=0
    print(k['(p,q)x(P,Q)'][f_aic])
    
    print(k)
    best_model = SARIMAX(train, order=(k['(p,q)x(P,Q)'][f_aic][0], 1, k['(p,q)x(P,Q)'][f_aic][1]), seasonal_order=(k['(p,q)x(P,Q)'][f_aic][2], 1, k['(p,q)x(P,Q)'][f_aic][2], 4)).fit(dis=-1)
    print(best_model.summary())
    pred= best_model.fittedvalues
    forecast = best_model.predict(start=train.shape[0], end=train.shape[0] + period)
    forecast1 = np. concatenate((forecast,pred),axis = 0)
    forecast1.std()
    predi= scaler.inverse_transform(forecast1.reshape(-1, 1))    
    return predi[len(predi)-period:]


# In[19]:


# pred_high=forecast(df['High'],7)


# In[23]:


# pred_low=forecast(df['Low'],7)


# In[24]:


# pred_close=forecast(df['Close'],7)


# In[25]:


# pred_open=forecast(df['Open'],7)


# In[26]:





# In[27]:





# In[28]:



    
    


# In[29]:





# In[30]:




# In[31]:



























# In[ ]:




