# -*- coding: utf-8 -*-
"""
Created on Fri May 27 14:59:25 2022

@author: umium
"""

import pandas as pd
import os
import numpy as np
from tensorflow.keras.models import load_model
from modules import ExploratoryDataAnalysis

#%% PATH
NEW_CUSTOMERS_PATH = os.path.join(os.getcwd(), 'new_customers.csv')
LE_PATH_MARRIED = os.path.join(os.getcwd(), 'saved_model','married.pkl')
LE_PATH_GRADUATED = os.path.join(os.getcwd(), 'saved_model','graduated.pkl')
LE_PATH_PROFESSION = os.path.join(os.getcwd(), 'saved_model','profession.pkl')
LE_PATH_SSCORE = os.path.join(os.getcwd(), 'saved_model','sscore.pkl')
KNN_PATH = os.path.join(os.getcwd(), 'saved_model','knn.pkl')
MMS_PATH = os.path.join(os.getcwd(), 'saved_model','mms.pkl')
OHE_PATH = os.path.join(os.getcwd(), 'saved_model','ohe.pkl')
MODEL_SAVE_PATH = os.path.join(os.getcwd(),'model.h5')
SAVE_RESULT = os.path.join(os.getcwd(),'new_customers.csv')

#%% EDA
#Step 1) Data loading
df = pd.read_csv(NEW_CUSTOMERS_PATH)

#Step 2) Data inspection & data visualization
df.info()

#Step 3) Data cleaning
#to label encode the categorical columns after clean the nan values using notnull approach
#load model
eda = ExploratoryDataAnalysis()
eda.label_encoder_load(df['Ever_Married'], LE_PATH_MARRIED)
eda.label_encoder_load(df['Graduated'], LE_PATH_GRADUATED)
eda.label_encoder_load(df['Profession'], LE_PATH_PROFESSION)
eda.label_encoder_load(df['Spending_Score'], LE_PATH_SSCORE)

#reposition
cat_clms = df[['Ever_Married','Graduated','Profession','Spending_Score']]
con_clms = df[['Age','Family_Size']]

#combine cat_clms and con_clms
df1 = pd.concat([cat_clms, con_clms], axis=1)

#filling nan value using KNN imputer approach and load pickle
df1_imputed = eda.KNNimputer_load(df1, KNN_PATH)

#%%
#Step 4) Data preprocessing
df1_scaled = eda.MinMaxScaler_load(df1_imputed, MMS_PATH)

#features and target
features = df1_scaled

#%% Model loading and model evaluation
model = load_model(MODEL_SAVE_PATH)

predicted = []
predicted = model.predict(features)

y_true = np.argmax(features, axis=1)
predicted = np.argmax(predicted, axis=1)

#%% Data update
df.to_csv(SAVE_RESULT, index=False)
