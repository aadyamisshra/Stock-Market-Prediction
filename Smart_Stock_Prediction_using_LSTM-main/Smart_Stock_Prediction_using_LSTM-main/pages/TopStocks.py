# To find the right stock we are calculating the value of the sum of predicted stock prices - the present day price

import streamlit as st
import numpy as np
from sklearn.preprocessing import MinMaxScaler
st.set_page_config(page_title="List of Stocks to invest on: ", page_icon="📈")
st.markdown("Working model for Smart Stock Prediction")
st.sidebar.header("Listing of the Top performing Stocks:")


st.write(
    """This page shows the graphs of the following stocks and post prediction, 
    tells which are the top 3 stocks to invest on. 
    Enjoy!"""
)

import numpy as np, pandas as pd, matplotlib.pyplot as plt
import pandas_datareader as data
import yfinance as yfin
from keras.models import load_model
import streamlit as st

start = '2016-01-01'
end = '2023-2-25'

#Used for plotting future prediction data
day_new = np.arange(0,100)
day_pred = np.arange(100,130)


st.write("""
Starting Date:  2016-01-01
Ending Date:    2023-02-25
""")

st.subheader("Following are the stocks taken for predictions")
st.write("""
   
    1. AAPL: Apple Inc.
    2. TSLA: Tesla, Inc.
    3. AMZN: Amazon.com, Inc.
    4. TATAMOTORS.NS: Tata Motors Limited (on the NSE)
    5. ^RUT: Russell 2000 Index
    6. ADANIENT.NS: Adani Enterprises Limited (on the NSE)
    7. ^NSEI: Nifty 50 Index
    8. ^N225: Nikkei 225 Index
    9. ^NSEBANK: Nifty Bank Index

""")

options = [ 'TSLA','AAPL', 'AMZN','TATAMOTORS.NS','^RUT','ADANIENT.NS','^NSEI','^N225','^NSEBANK']
#using pre-trained model
model = load_model('tesla_model.h5')
vals = dict()


for option in options:
    
    df = yfin.download(option,start,end)
 
    data_training = pd.DataFrame(df['Close'][0:int(len(df)*0.70)]) #70% of data
    data_testing = pd.DataFrame(df['Close'][int(len(df)*0.70):])   #30% of data
    
    # scaled = scaler.scale_
    # scale_factor = 1/scaled[0]
    #Prediction of next data
    scaler = MinMaxScaler(feature_range=(0,1))
    
    data_test = scaler.fit_transform(data_testing) #only do once in code
    # data_training_arr = scaler.fit_transform(data_training)
    scaled = scaler.scale_
    scale_factor = 1/scaled[0]
    x_input = data_test
    start = (len(data_testing) - 100)
    x_input = x_input[start:].reshape(1,-1)


    temp_input = list(x_input)
    temp_input = temp_input[0].tolist()

    from numpy import array
    lst_output = []
    n_steps = 100
    i =0
    while(i<30):
        if(len(temp_input)>100):
            x_input = np.array(temp_input[1:])
            # print("{} day input {}".format(i,x_input))
            x_input = x_input.reshape(1,-1)
            x_input = x_input.reshape((1,n_steps,1))
            y_hat = model.predict(x_input,verbose=2)
            # print("{} day output {}".format(i,y_hat))
            temp_input.extend(y_hat[0].tolist())
            temp_input =temp_input[1:]
            lst_output.extend(y_hat.tolist())
            i=i+1
        else:
            x_input = x_input.reshape((1,n_steps,1))
            y_hat = model.predict(x_input,verbose =0) # verbose =1,shows a bar when prediction is done
            # print(y_hat[0])
            temp_input.extend(y_hat[0].tolist())
            # print(len(temp_input))
            lst_output.extend(y_hat.tolist())
            i=i+1
        #Plotting the stock trend graph

    length = len(df.Close)
    fig = plt.figure(figsize=(12,6))
    plt.plot(day_new,(df.Close[(length-100):length]),'b',label = "Original Price")
    plt.plot(day_pred,scaler.inverse_transform(lst_output),'r',label = "Predicted Price")
    plt.ylabel("Price")
    plt.xlabel("Days(0-100) and Next 30 Days")
    plt.title(f"Stock: {option}")
    plt.legend()
    st.pyplot(fig)

    lst = np.array(lst_output)
    value = data_test[-1]
    value = int(sum(lst)-(30*value))
    vals[option]= value
    
#Vals is a dictionary containing the stock names as KEYS and calculated stock efficiency in 30 days as VALUES. 
    
vals =sorted(vals.items(), key=lambda x: x[1], reverse=True)
st.subheader("The top 3 Stocks to invest today are:")
for i in range(3):
    key,value = vals[i]
    st.write(f"{i+1} -> {key}")



