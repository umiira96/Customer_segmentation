# -*- coding: utf-8 -*-
"""
Created on Wed May 18 15:15:28 2022

@author: umium
"""

import pickle
import numpy as np
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.impute import KNNImputer
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Dense, BatchNormalization, Dropout
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score

class ExploratoryDataAnalysis():
    def label_encoder(self,data,path):
        le = LabelEncoder()
        data[data.notnull()] = le.fit_transform(data[data.notnull()])
        
        pickle.dump(le,open(path, 'wb'))
        return data
    
    def label_encoder_load(self,data,path):
        le = LabelEncoder()
        data[data.notnull()] = le.fit_transform(data[data.notnull()])
        
        path = pickle.load(open(path,'rb'))
        return data
    
    def KNNimputer(self,data, path):
        knn = KNNImputer()
        data = knn.fit_transform(data)
        
        pickle.dump(knn,open(path, 'wb'))
        return data
    
    def KNNimputer_load(self,data, path):
        knn = KNNImputer()
        data = knn.fit_transform(data)
        
        path = pickle.load(open(path,'rb'))
        return data
    
    def MinMaxScaler(self,data,path):
        mms = MinMaxScaler()
        data = mms.fit_transform(data)
        
        pickle.dump(mms,open(path, 'wb'))
        return data
    
    def MinMaxScaler_load(self,data,path):
        mms = MinMaxScaler()
        data = mms.fit_transform(data)
        
        path = pickle.load(open(path,'rb'))
        return data
    
    def OneHotEncoder(self,data,path):
        ohe = OneHotEncoder(sparse=False)
        data = ohe.fit_transform(np.expand_dims(data, axis=-1))
        
        pickle.dump(ohe,open(path, 'wb'))
        return data

class ModelCreation():
    def model_layers(self, input_shape, nb_nodes, output_shape):

        model = Sequential()
        model.add(Input(shape=(input_shape)))
        model.add(Dense(nb_nodes, activation='relu'))
        model.add(Dropout(0.2))
        model.add(BatchNormalization())
        model.add(Dense(nb_nodes))
        model.add(Dropout(0.2))
        model.add(BatchNormalization())
        model.add(Dense(output_shape))
        
        
        model.compile(optimizer='adam',
                      loss='mse',
                      metrics='mse') 
        
        model.summary()
        return model

class ModelEvaluation():
    def report_metrics(self,y_true,predicted):
        print(classification_report(y_true,predicted))
        print(confusion_matrix(y_true,predicted))
        print(accuracy_score(y_true,predicted))
