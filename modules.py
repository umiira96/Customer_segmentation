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
        model.add(Dense(128, activation='relu', 
                        input_shape=[60,1]))
        model.add(Dropout(0.2))
        model.add(BatchNormalization())
        model.add(Dense(128))
        model.add(Dropout(0.2))
        model.add(BatchNormalization())
        model.add(Dense(1))

class ModelEvaluation():
    def report_metrics(self,y_true,predicted):
        print(classification_report(y_true,predicted))
        print(confusion_matrix(y_true,predicted))
        print(accuracy_score(y_true,predicted))
        #%%
        
if __name__ == '__main__':
    
    TRAIN_PATH = os.path.join(os.getcwd(), 'train.csv')
    NEW_CUSTOMERS_PATH = os.path.join(os.getcwd(), 'new_customers.csv')
    PATH = os.path.join(os.getcwd(),'logs')
    MODEL_SAVE_PATH = os.path.join(os.getcwd(),'model.h5')

    
    