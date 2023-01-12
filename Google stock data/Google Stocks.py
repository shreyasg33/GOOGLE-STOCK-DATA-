#!/usr/bin/env python
# coding: utf-8

# ## Importing libraries

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# ## Reading the csv file

# In[2]:


df=pd.read_csv(r'C:\Users\HAPPY BIRTHDAY\Desktop\projects\Google stock data\Google.csv')


# ## Data visualization

# In[3]:


df.head()


# In[4]:


df.describe()


# In[5]:


df.info()


# In[7]:


df.isnull().sum()


# In[8]:


df['Date'] = pd.to_datetime(df['Date'])


# In[9]:


df = df.set_index('Date')


# In[10]:


df.tail()


# In[13]:


corr = df.corr()


# In[14]:


corr.style.background_gradient(cmap = 'coolwarm')


# In[16]:


df['MA30'] = df['Adj Close'].rolling(window = 30).mean()
df['MA100'] = df['Adj Close'].rolling(window = 100).mean()


# In[17]:


def get_RSI(df, column = 'Adj Close', time_window = 14):
    # Return the RSI indicator for the specified time window

    diff = df[column].diff(1)

    # This preservers dimensions off diff values. 
    up_chg = 0 * diff
    down_chg = 0 * diff

    # Up change is equal to the positive difference, otherwise equal to zero
    up_chg[diff >0] = diff[diff > 0]

    # Down change is equal to negative difference, otherwise equal to zero
    down_chg[diff < 0] = diff[diff <0]
    
     # We set com = time_window-1 so we get decay alpha =1/time_window.
    up_chg_avg = up_chg.ewm(com=time_window -1,min_periods = time_window).mean()
    down_chg_avg = down_chg.ewm(com = time_window -1, min_periods = time_window).mean()

    RS = abs(up_chg_avg/down_chg_avg)
    df['RSI'] = 100 - 100 / (1 + RS)

    return df


# In[18]:


get_RSI(df)


# In[36]:


plt.figure(figsize = (15,5))
plt.plot(df['MA30'])
plt.axhline(y = 500,color = 'red')
plt.axhline(y = 2500,color = 'red')




# ## EDA

# In[37]:


plt.figure(figsize = (15,5))
plt.plot(df['MA100'])
plt.axhline(y = 500,color = 'red')
plt.axhline(y = 2500,color = 'red')



# In[66]:


plt.figure(figsize = (15,5))
plt.plot(df['RSI'])
plt.axhline(y = 80,color = 'red')
plt.axhline(y = 30,color = 'red')
plt.title('Relative strength index')


# In[39]:


data = pd.DataFrame()
data['Goog'] = df['Adj Close']
data['MA30'] = df['MA30']
data['MA100'] = df['MA100']
data


# In[48]:


train_df = df['High'].iloc[:-4]

X_train = []
y_train = []

for i in range(2, len(train_df)):
  X_train.append(train_df[i-2:i])
  y_train.append(train_df[i])


# In[49]:


import math
train_len = math.ceil(len(train_df)*0.8)
train_len


# In[47]:


from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dense, Activation, Dropout, TimeDistributed
from tensorflow.keras.models import Sequential


# In[50]:


X_train, y_train= np.array(X_train), np.array(y_train)
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))


# In[54]:


model=Sequential()
model.add(LSTM(50,activation='relu', input_shape=(X_train.shape[1],1)))
model.add(Dense(25))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')
model.summary()
model.fit(X_train, y_train, epochs=200, batch_size=100)


# In[55]:


losse = pd.DataFrame(model.history.history)
losse[['loss']].plot()


# In[56]:


test_data = train_df[train_len-2:]
X_val=[]
Y_val=[] 

for i in range(2, len(test_data)):
    X_val.append(test_data[i-2:i])
    Y_val.append(test_data[i])


# In[57]:


X_val, Y_val = np.array(X_val), np.array(Y_val)
X_val = np.reshape(X_val, (X_val.shape[0], X_val.shape[1],1))
prediction = model.predict(X_val)


# In[58]:


from sklearn.metrics import mean_squared_error
# Know the model error accuracy | the model accuracy 
lstm_train_pred = model.predict(X_train)
lstm_valid_pred = model.predict(X_val)
print('Train rmse:', np.sqrt(mean_squared_error(y_train, lstm_train_pred)))
print('Validation rmse:', np.sqrt(mean_squared_error(Y_val, lstm_valid_pred)))


# In[59]:


valid = pd.DataFrame(train_df[train_len:])
valid['Predictions']=lstm_valid_pred 
plt.figure(figsize=(16,8))
plt.plot(valid[['High','Predictions']])
plt.legend(['Validation','Predictions'])
plt.show()


# In[60]:


variance = []
for i in range(len(valid)):
  
  variance.append(valid['High'][i]-valid['Predictions'][i])
variance = pd.DataFrame(variance)
variance.describe()


# In[61]:


train = train_df[:train_len]
valid = pd.DataFrame(train_df[train_len:])
valid['Predictions']=lstm_valid_pred

plt.figure(figsize=(16,8))
plt.title('Model LSTM')
plt.xlabel('Date')
plt.ylabel('Google Price USD')
plt.plot(train)
plt.plot(valid[['High','Predictions']])
plt.legend(['Train','Val','Predictions'])
plt.show()


# In[ ]:




