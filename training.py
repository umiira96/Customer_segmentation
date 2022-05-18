# -*- coding: utf-8 -*-
"""
Created on Wed May 18 14:47:09 2022

@author: umium
"""
import pandas as pd
import os
import seaborn as sns
import numpy as np
import datetime
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, BatchNormalization, Dropout
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.callbacks import TensorBoard, EarlyStopping
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score


TRAIN_PATH = os.path.join(os.getcwd(), 'train.csv')
NEW_CUSTOMERS_PATH = os.path.join(os.getcwd(), 'new_customers.csv')
PATH = os.path.join(os.getcwd(),'logs')
MODEL_SAVE_PATH = os.path.join(os.getcwd(),'model.h5')


#%% EDA
#Step 1) Data loading
train = pd.read_csv(TRAIN_PATH)
test = pd.read_csv(NEW_CUSTOMERS_PATH)

train.info()
test.info()

#to concat train and test data
train['train_y_n']=1
test['train_y_n']=0
train_test = pd.concat([train,test], axis=0)

#to visualize misiing values
sns.heatmap(train_test.isna())

train_test.info()

sns.heatmap(train_test.corr(),annot=True)

#treat missing values
train_test.isnull().sum()


#Ever married
train_test[train_test['Ever_Married'].isnull()]['Family_Size'].value_counts()

train_test['Ever_Married']=train_test['Ever_Married'].fillna('Yes')

train_test=pd.get_dummies(train_test,prefix='Married',columns=['Ever_Married'],drop_first=True)

#Graduated
train_test['Graduated']=train_test['Graduated'].fillna('Yes')

train_test=pd.get_dummies(train_test,prefix='Graduated',columns=['Graduated'],drop_first=True)

#Profession
train_test['Profession'].fillna('Unknown',inplace=True)

train_test['Profession']=train_test['Profession'].astype('str')

le = LabelEncoder()
train_test['Profession_le']=le.fit_transform(train_test['Profession'])
train_test.drop('Profession',axis=1,inplace=True)

#Work_Experience
train_test['Work_Experience'].fillna(train_test['Work_Experience'].median(),inplace=True)

#Spending_Score
train_test.loc[train_test['Spending_Score']=='Low','Spending_Score']=1
train_test.loc[train_test['Spending_Score']=='Average','Spending_Score']=2
train_test.loc[train_test['Spending_Score']=='High','Spending_Score']=3
train_test['Spending_Score']=train_test['Spending_Score'].astype('int')

#Family_Size
train_test['Family_Size'].fillna(round(train_test['Family_Size'].median()),inplace=True)

#Var_1
train_test['Var_1'].fillna('Cat_6',inplace=True)
train_test['Var_1']=train_test['Var_1'].apply(lambda x:x[-1])
train_test['Var_1']=train_test['Var_1'].astype('int')

#Gender
train_test['Gender']=le.fit_transform(train_test['Gender'])

train_test.isnull().sum()

#separate train and test data
df = pd.DataFrame(train_test)

X_train = df['train_y_n']==1
X_train = df[X_train]


X_test = df['train_y_n']==0
X_test = df[X_test]

X_train['Segmentation'] = le.fit_transform(X_train['Segmentation'])

#%%

X_train = X_train.iloc[: , 1:]
X_test = X_test.iloc[: , 1:]

mms = MinMaxScaler()
x_train_scaled = mms.fit_transform(X_train)
x_test_scaled = mms.transform(X_test)

#training dataset
x_train = []
y_train = []

for i in range(100,len(X_train)):
    x_train.append(x_train_scaled[i-100:i,1])
    y_train.append(x_train_scaled[i,1])

x_train = np.array(x_train)
y_train = np.array(y_train)

#Testing dataset
window_size = 100
#scaled_X_test, scaled_X_train both in array
temp = np.concatenate((x_train_scaled, x_test_scaled))
length_window = window_size+len(x_test_scaled)
temp = temp[-length_window:]

x_test = []
y_test = []
for i in range(window_size,len(temp)):
    x_test.append(temp[i-window_size:i,1])
    y_test.append(temp[i,1])
    
x_test = np.array(x_test)
y_test = np.array(y_test)

x_train = np.expand_dims(x_train, axis=-1)
x_test = np.expand_dims(x_test, axis=-1)

#%% Model creation

input_dim = x_train.shape[1]

model = Sequential()
model.add(Dense(128, activation='tanh', 
                input_shape=[100]))
model.add(Dropout(0.2))
model.add(BatchNormalization())
model.add(Dense(128))
model.add(Dropout(0.2))
model.add(BatchNormalization())
model.add(Dense(1))


model.compile(optimizer='adam',
              loss='mse',
              metrics='mse') 



log_files = os.path.join(PATH,
                         datetime.datetime.now().strftime('%Y%m%d-%H%M%S'))

# Tensorboard and earlystopping callback
tensorboard_callback = TensorBoard(log_dir=log_files, histogram_freq=1)
early_stopping_callback = EarlyStopping(monitor='val_loss', patience=3)

hist = model.fit(x_train,y_train,epochs=50,validation_data=(x_test,y_test),
                 callbacks=tensorboard_callback)
                 
print(hist.history.keys())

plt.figure()
plt.plot(hist.history['loss'])
plt.xlabel('epochs')
plt.ylabel('training loss')
plt.show()

plt.figure()
plt.plot(hist.history['loss'])
plt.plot(hist.history['val_loss'])
plt.show()

plt.figure()
plt.plot(hist.history['mse'])
plt.plot(hist.history['val_mse'])
plt.show()



#%%
y_pred = model.predict(x_test)
y_true = y_test

print(classification_report(y_true,y_pred))
print(confusion_matrix(y_true,y_pred))
print(accuracy_score(y_true,y_pred))


#%% model deployment
model.save(MODEL_SAVE_PATH)
