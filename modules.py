# -*- coding: utf-8 -*-
"""
Created on Wed May 18 15:15:28 2022

@author: umium
"""

import os
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, BatchNormalization, Dropout
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score


class ModelCreation():
    
    def layers(self):

        model = Sequential()
        model.add(Dense(128, activation='tanh', 
                        input_shape=[100]))
        model.add(Dropout(0.2))
        model.add(BatchNormalization())
        model.add(Dense(128))
        model.add(Dropout(0.2))
        model.add(BatchNormalization())
        model.add(Dense(1))

class ModelEvaluation():
    def report_metrics(self,y_true,y_pred):
        print(classification_report(y_true,y_pred))
        print(confusion_matrix(y_true,y_pred))
        print(accuracy_score(y_true,y_pred))
        #%%
        
if __name__ == '__main__':
    
    TRAIN_PATH = os.path.join(os.getcwd(), 'train.csv')
    NEW_CUSTOMERS_PATH = os.path.join(os.getcwd(), 'new_customers.csv')
    PATH = os.path.join(os.getcwd(),'logs')
    MODEL_SAVE_PATH = os.path.join(os.getcwd(),'model.h5')

    
    