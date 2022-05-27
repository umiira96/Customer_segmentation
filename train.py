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
import missingno as msno
from modules import ExploratoryDataAnalysis, ModelCreation, ModelEvaluation
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import TensorBoard, EarlyStopping

#%% PATH
TRAIN_PATH = os.path.join(os.getcwd(), 'train.csv')
LE_PATH_MARRIED = os.path.join(os.getcwd(), 'saved_model','married.pkl')
LE_PATH_GRADUATED = os.path.join(os.getcwd(), 'saved_model','graduated.pkl')
LE_PATH_PROFESSION = os.path.join(os.getcwd(), 'saved_model','profession.pkl')
LE_PATH_SSCORE = os.path.join(os.getcwd(), 'saved_model','sscore.pkl')
KNN_PATH = os.path.join(os.getcwd(), 'saved_model','knn.pkl')
MMS_PATH = os.path.join(os.getcwd(), 'saved_model','mms.pkl')
OHE_PATH = os.path.join(os.getcwd(), 'saved_model','ohe.pkl')
PATH = os.path.join(os.getcwd(),'logs')
MODEL_SAVE_PATH = os.path.join(os.getcwd(),'model.h5')

#%% EDA
#Step 1) Data loading
df = pd.read_csv(TRAIN_PATH)

#Step 2) Data inspection & data visualization
df.info()

sns.countplot(df['Gender'], hue=df['Segmentation'])
sns.countplot(df['Ever_Married'], hue=df['Segmentation'])
sns.countplot(df['Graduated'], hue=df['Segmentation'])
sns.countplot(df['Profession'], hue=df['Segmentation'])
sns.countplot(df['Spending_Score'], hue=df['Segmentation'])
#gender and var_1 columns shows weak trends

#to visualize the missing value
msno.matrix(df)
#work_experience column shows many missing values

#Step 3) Data cleaning
#drop clms with weak trends(gender,var_1,work_experience)
cat_clms = df[['Ever_Married','Graduated','Profession','Spending_Score']]
con_clms = df[['Age','Family_Size']]

#to label encode the categorical columns after clean the nan values using notnull approach
#save the model using pickle
eda = ExploratoryDataAnalysis()
eda.label_encoder(df['Ever_Married'], LE_PATH_MARRIED)
eda.label_encoder(df['Graduated'], LE_PATH_GRADUATED)
eda.label_encoder(df['Profession'], LE_PATH_PROFESSION)
eda.label_encoder(df['Spending_Score'], LE_PATH_SSCORE)

#reposition
cat_clms = df[['Ever_Married','Graduated','Profession','Spending_Score']]
con_clms = df[['Age','Family_Size']]

#combine cat_clms and con_clms
df1 = pd.concat([cat_clms, con_clms], axis=1)

#filling nan value using KNN imputer approach
df1_imputed = eda.KNNimputer(df1, KNN_PATH)

#%%
#Step 4) Data preprocessing
df1_scaled = eda.MinMaxScaler(df1_imputed, MMS_PATH)

#features and target
features = df1_scaled
target = df['Segmentation']

target = eda.OneHotEncoder(target, OHE_PATH)

#train test split
X_train,x_test,y_train,y_test = train_test_split(features,target,
                                                 test_size=0.3, 
                                                 random_state=42 )

#%% Model creation

input_dim = X_train.shape[1]

mc = ModelCreation()
model = mc.model_layers(input_shape=6, nb_nodes=128, output_shape=4)

log_files = os.path.join(PATH,
                         datetime.datetime.now().strftime('%Y%m%d-%H%M%S'))

# Tensorboard and earlystopping callback
tensorboard_callback = TensorBoard(log_dir=log_files, histogram_freq=1)
early_stopping_callback = EarlyStopping(monitor='val_loss', patience=10)

hist = model.fit(X_train,y_train,epochs=100,validation_data=(x_test,y_test),
                 callbacks=[tensorboard_callback, early_stopping_callback])
                 
print(hist.history.keys())

#%% Model evaluation

predicted = []
predicted = model.predict(x_test)

y_true = np.argmax(y_test, axis=1)
predicted = np.argmax(predicted, axis=1)

me = ModelEvaluation()
me.report_metrics(y_true,predicted)


#%% Model deployment
model.save(MODEL_SAVE_PATH)
