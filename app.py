import datetime
import time
import pandas as pd
import streamlit as st
from stock_market import *
import yfinance as yf
from PIL import Image
   
f_high=[]
f_low=[]
f_open=[]
f_close=[]
date_c=[]
import base64
def add_bg_from_local(image_file):
    with open(image_file, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read())
    st.markdown(
    f"""
    <style>
    .stApp {{
        background-image: url(data:image/{"png"};base64,{encoded_string.decode()});
        background-size: cover
    }}
    </style>
    """,
    unsafe_allow_html=True
    )


def main():
    st.set_page_config(page_title='Estimator using SARIMAX ',page_icon="🕗", layout="centered", initial_sidebar_state="auto", menu_items=None)
    im ="Stock_bb.png"
    add_bg_from_local(im)    
    st.title("Asset Price Estimator using time series ")
    Header= '<p style="font-family:Georgia; color:maroon; font-size: 25px;">Enter the company or asset ticker to the get stock prediction for specified days. You can also get the trade strtegy using technical analysis tools 💹</p>'
    st.warning('Please run the app only after market in which respective asset is listed closes for accurate results', icon="⚠️")
    st.markdown(Header,unsafe_allow_html=True)
    com=st.text_input('Ticker symbol of the company')
    if not com :
        st.info('Enter ticker ↩️')
        st.stop()
    d = st.slider('How  many days to be predicted', 1, 10)
    time.sleep(3)    
    df=yf.download(tickers=com,period='6mo',interval='1d')   
    df=df.reset_index()
    df.columns=['Date','Open','High', 'Low' ,'Close','Adj Close','Volume']
    st.write('Predict',com, 'for' , d , 'days')
    #st.write(len(vwap))
    try:
        ma=get_macd(df['Close'], 26,12,9)
        HA=Heinki_Ashi(df['Open'],df['High'],df['Low'],df['Close'])
        vwap=VWAP(df['High'],df['Low'],df['Close'],df['Volume'],ma['Trend'])

        bb=bollinger_band(df['High'],df['Low'],df['Close'],df['Volume'])
        ha=HA_strategy(df['Open'],df['High'],df['Close'],df['Low'],HA)
    except:
        st.warning('Please check spelling of ticker', icon="⚠️")
        st.stop()


    st.write(cand_plot(df['Date'],df['Open'],df['High'],df['Close'],df['Low'],com))
    if "load_state" not in st.session_state:
        st.session_state.load_state = False
    st.write('Prediction start')

                        
    def descion(signal,df_f):
        if signal[len(signal)-1]=='Buy':
            st.write('Buy at',(datetime.date.today()+datetime.timedelta(days=1)),'for' ,df_f['Low'][1])
        elif signal[len(signal)-1]=='Sell':
            st.write('Sell at',(datetime.date.today()+datetime.timedelta(days=1)),'for' ,df_f['High'][1])
        else:
            st.write('Hold, prices are high',df_f['High'][1],'low',df_f['Low'][1],(datetime.date.today()+datetime.timedelta(days=1)) )
    

    if st.button('Desicion to buy or sell and prediction') or st.session_state.load_state : #or st.session_state.load_state:
        @st.cache(ttl=24*3600)  
        def comp():
            with st.spinner('Analysing...'):
                st.session_state.load_state = False
                pred_high=forecast(df['High'],d)
                pred_low=forecast(df['Low'],d)
                pred_open=forecast(df['Open'],d)
                pred_close=forecast(df['Close'],d)
                f_high.append(df['High'][len(df['High'])-1]) 
                f_low.append(df['Low'][len(df['Low'])-1])
                f_open.append(df['Open'][len(df['Open'])-1])
                f_close.append(df['Close'][len(df['Close'])-1])
                date_c.append(df['Date'][len(df['Close'])-1])
            for i in range(0,d):
                f_high.append(float(max(f_high[i]+pred_high[i],f_close[i]+pred_close[i],f_open[i]+pred_open[i])))
                f_low.append(float(min(f_close[i]+pred_close[i],f_open[i]+pred_open[i],f_low[i]+pred_low[i])))
                f_open.append(float(f_open[i]+pred_open[i]))
                f_close.append(float(f_close[i]+pred_close[i]))
                date_c.append(date_c[0]+datetime.timedelta(days=i))
            f_pre=pd.DataFrame(np.column_stack([date_c,f_open,f_high,f_low,f_close]), 
                                columns=['Date','Open','High','Low','Closes',])
            return f_pre                    
        df_f=comp()
        st. success('Predcition Complete',icon="✅")
        st.write("prices are")
        st.write(df_f)
        Macd= '<p style="font-family:Cambria; color:RoyalBlue; font-size: 15px;">Moving average convergence/divergence (MACD, or MAC-D) is a trend-following momentum indicator that shows the relationship between two exponential moving averages (EMAs) of a security’s price</p>'
        st.markdown(Macd,unsafe_allow_html=True)
        descion(ma['MACD_signal'],df_f)
        BB= '<p style="font-family:Baskerville; color:DarkCyan; font-size: 15px;">A Bollinger Band is a technical analysis tool defined by a set of trendlines plotted two standard deviations (positively and negatively) away from a simple moving average (SMA) of a security`s price</p>'
        st.markdown(BB,unsafe_allow_html=True)    
        descion(bb['BB_signal'],df_f)
        vs= '<p style="font-family:Cambria; color:RoyalBlue; font-size: 15px;">VWAP is a trading benchmark that represents the average price a security has traded at throughout the day, based on both volume and price</p>'
        st.markdown(vs,unsafe_allow_html=True)
        descion(vwap['VWAP_signal'],df_f)
        hs='<p style="font-family:Baskerville; color:DarkCyan; font-size: 15px;">Heikin Ashi candlesticks are calculated based on averages, the candlesticks will have smaller shadows and helps in reducing noises </p>'
        st.markdown(hs,unsafe_allow_html=True)
        descion(ha['HA_signal'],df_f)
    def profit(signal):

        port=1000
        no=0
        port1=1000
        for i in range(len(signal)-1):
            if  signal.iloc[i]  =='Buy' and port>0 :
                no=(port)/(df['Open'].iloc[i+1])
                port=0
            elif signal.iloc[i]  =='Sell' and no>0:
                port=no*(df['Open'].iloc[i+1])
                port=port-port*(0.012)
                no=0
        if  no*(df['Close'].iloc[len(signal)-1])!=0 :        
            st. write((no*(df['Close'].iloc[len(signal)-1])-port1)*100/port1)
        if  port!=0:
            st. write((port-port1)*100/port1)
        elif port==port1 or  no*(df['Close'].iloc[len(signal)-1])==port1 : 
            st. write('0')       
    if st.button('Profit for past 14 days'):#  or st.session_state.load_state:
        st.session_state.load_state = False
        st. write("Profit if HA strategy is followed for past 14 days")
        profit(ha['HA_signal'][len(ha['HA_signal'])-15:])
        st. write("Profit if MACD strategy is followed for past 14 days")
        profit(ma['MACD_signal'][len(ma['MACD_signal'])-15:])
        st. write("Profit if bollinger band strategy is followed for past 14 days")
        profit(bb['BB_signal'][len(bb['BB_signal'])-15:])
        st. write("Profit if VWAP strategy is followed for past 14 days")
        profit(vwap['VWAP_signal'][len(vwap['VWAP_signal'])-15:])
    
                          
if __name__==main():
    main()
