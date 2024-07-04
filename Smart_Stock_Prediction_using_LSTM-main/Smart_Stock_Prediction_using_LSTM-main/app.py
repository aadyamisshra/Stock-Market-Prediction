import numpy as np, pandas as pd, matplotlib.pyplot as plt
import pandas_datareader as data
import datetime
import yfinance as yfin
from keras.models import load_model
import streamlit as st

start = '2016-01-01'
end = '2023-2-25'
st.sidebar.success("Select a demo above.")

# ADANIENT.NS, AMZN, ^RUT, TATASTEEL.NS,YM=F
st.title("Stock Trend Prediction")
option = st.selectbox(
    'Select a stock to analyze:',
    ('AAPL', 'TSLA', 'AMZN','TATAMOTORS.NS','^RUT','ADANIENT.NS','^NSEI','^BSESN','^BSESN','^BSEI','^NSEI','^BSEI','^N225','^NSEBANK','^BSEBANK','^BSESN'))

# option= st.text_input('Enter stock ticker','AAPL') #user input

df = yfin.download(option,start,end)

st.subheader('Data from 2016-2023')
st.write(df.describe())

st.subheader("Closing Price vs. Time Chart")
fig = plt.figure(figsize=(12,6))
plt.plot(df['Close'])
plt.xlabel('Time')
plt.ylabel('Closing Price')
st.pyplot(fig)

st.subheader("Closing Price vs. Time Chart with 100 Days and 200 Days Moving Avg.")
ma100 = df.Close.rolling(100).mean()
ma200 = df.Close.rolling(200).mean()

fig = plt.figure(figsize=(12,6))
plt.plot(ma100,'b')
plt.plot(ma200,'r')
plt.plot(df.Close)
st.pyplot(fig)

data_training = pd.DataFrame(df['Close'][0:int(len(df)*0.70)]) #70% of data
data_testing = pd.DataFrame(df['Close'][int(len(df)*0.70):])   #30% of data
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0,1))

data_training_arr = scaler.fit_transform(data_training)

#using pre-trained model
model = load_model('tesla_model.h5')

#testing part
past_100_days = data_training.tail(100)
final_df = past_100_days.append(data_testing,ignore_index = True)
x_test = []
y_test = []
input_data = scaler.fit_transform(final_df)
for i in range(100,input_data.shape[0]):
    x_test.append(input_data[i-100:i])
    y_test.append(input_data[i,0]) #0 is the closing price columns

x_test,y_test  = np.array(x_test),np.array(y_test)
y_predicted = model.predict(x_test)
scaled = scaler.scale_
scale_factor = 1/scaled[0]
y_predicted = y_predicted * scale_factor
y_test = y_test * scale_factor

#final graph
st.subheader('predictions vs. original')
fig2 = plt.figure(figsize = (12,6))
plt.plot(y_test,'b',label = "Original Price")
plt.plot(y_predicted,'r',label = "Predicted Price")
plt.xlabel('Days (2016-01-01 to 2023-2-25)')
plt.ylabel('Price')
plt.legend()
st.pyplot(fig2)


#Prediction of next data
st.subheader('Next 30 days prediction')
data_test = scaler.fit_transform(data_testing) #only do once in code
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
        y_hat = model.predict(x_input,verbose=0)
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
day_new = np.arange(1,101)
day_pred = np.arange(101,131)   
length = len(df.Close)

fig3 = plt.figure(figsize=(12,6))
plt.plot(day_new,(df['Close'][(length-100):length]),'b',label = "Original Price")
plt.plot(day_pred,scaler.inverse_transform(lst_output),'r',label = "Predicted Price")
plt.legend()
plt.ylabel("Price")
plt.xlabel("Days(0-100) and Next 30 Days")
st.pyplot(fig3)

#plotting the original graph

start_new = '2016-01-01'
end_new = '2023-3-31'
st.subheader("The Original Graph")
df_new = yfin.download(option,start_new,end_new)
fig4 = plt.figure(figsize = (12,6))
plt.plot(df_new['Close'][length-100:])
plt.ylabel("Price")
plt.xlabel("Days(2016-01-01 to 2023-2-25) and Next 30 Days")
st.pyplot(fig4)



#use <streamlit run app.py> in terminal to run the code
#change environment or install all dependencies if the code does not run
#If you are working on conda env, you can also create a new env. and load dependencies in it.