#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 17 07:53:18 2019

@author: ramakrishna
"""

import numpy as np
import pandas as pd
import time



url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
iris = pd.read_csv(url, names=names)

 #shuffles the data along the row
iris1 = iris.values
iris
irisx = iris1
for i in range(len(irisx)):
        if irisx[i][4] == "Iris-setosa":
            irisx[i][4] = 0
        elif irisx[i][4] == "Iris-versicolor":
            irisx[i][4] = 1
        else:
            irisx[i][4] = 2
        for j in range(len(irisx[i])-1):
            irisx[i][j] = float(irisx[i][j])
            
np.random.shuffle(irisx)

iris_features1 = np.array(irisx[:,0:4],dtype = np.float64)
Trainfeatures1 = iris_features1[0:120,:]   #80% of the data set
Testfeatures1 = iris_features1[120:150,:] #20% of the data set
iris_labels1 = np.array(irisx[:,4:5], dtype = object)
iris_labels2 = iris_labels1[0:120,:]

def sigmoid (x):
    return 1/(1 + np.exp(-x))

#Derivative of Sigmoid Function
def derivatives_sigmoid(x):
    return x * (1 - x)


def standard(data):
    standardData = data.copy()
    
    rows = data.shape[0]
    cols = data.shape[1]

    for j in range(cols):
        sigma = np.std(data[:,j])
        mu = np.mean(data[:,j])

        for i in range(rows):
            standardData[i,j] = (data[i,j] - mu)/sigma

    return standardData

tf2 = standard(Trainfeatures1)


lr=0.01 #Setting learning rate
momentum = 0.9

HiddenNuerons = 4
InputNuerons = 4
OutputNuerons = 3

weight_h_layer1 = np.random.uniform(size=(InputNuerons,HiddenNuerons))
bias_h_layer1=np.random.uniform(size=(1,HiddenNuerons))
weight_h_layer2 = np.random.uniform(size=(HiddenNuerons,OutputNuerons))
bias_h_layer2=np.random.uniform(size=(1,OutputNuerons))
#bias_h_layer1 = np.zeros(HiddenNuerons)
#bias_h_layer2 = np.zeros(OutputNuerons)

input_data1 = tf2
Trainlabels1 = iris_labels1[0:120,:]
y1 = Trainlabels1

parameters = [weight_h_layer1, weight_h_layer2, bias_h_layer1, bias_h_layer2]


def nueralnet(input_data, y, epoch, weight_h_layer1, weight_h_layer2, bias_h_layer1, bias_h_layer2):
     for i in range(epoch):
          # forward - matrix multiply +bias
        input1 = np.dot(input_data, weight_h_layer1) + bias_h_layer1
        input2 = np.tanh(input1)
        
        # backward
        output1 = np.dot(input2, weight_h_layer2) + bias_h_layer2
        output2 = sigmoid(output1)
        
        error_h_layer2 = y - output2
    
        slope_h_layer2 = derivatives_sigmoid(output2)
        slope_h_layer1 = derivatives_sigmoid(input2)
    
        gradient_h_layer2 = error_h_layer2 * slope_h_layer2
        error_h_layer1 = gradient_h_layer2.dot(weight_h_layer2.T)
        gradient_h_layer1 = error_h_layer1 * slope_h_layer1
    
        weight_h_layer2 = weight_h_layer2 + (input2.T.dot( gradient_h_layer2) *lr)
        bias_h_layer2 = bias_h_layer2 + (np.sum(gradient_h_layer2, axis=0,keepdims=True) *lr)
        weight_h_layer1 = weight_h_layer1 + (input_data.T.dot(gradient_h_layer1) *lr)
        bias_h_layer1 = bias_h_layer1 + (np.sum(gradient_h_layer1, axis=0,keepdims=True) *lr)
        
        #entropy
        loss = -np.mean(y * np.log(output2) + (1-y) * np.log(1-output2))
        
        # loss
        t0 = time.clock()
    
        # classifying
        for (x,y), value in np.ndenumerate(output2):
            
       
            if(value<0.33):
                output2[x][y] = 0
            elif(0.33 < value < 0.66):
                output2[x][y] = 1
            else:
                output2[x][y] = 2
    
        #return output2
        return output2, loss, time.clock()-t0  
    
   

   
    #return  loss, (gradient_h_layer1, gradient_h_layer2, error_h_layer1, error_h_layer2)

nueralnet(input_data1, y1, 1000, *parameters)





def predict(input_data, h_layer1, h_layer2, bias_h_layer1, bias_h_layer2):
    input1 = np.dot(input_data, weight_h_layer1) + bias_h_layer1
    input2 = np.tanh(input1)

    output1 = np.dot(input2, weight_h_layer2) + bias_h_layer2
    output2 = sigmoid(output1)
    
    for (x,y), value in np.ndenumerate(output2):
       
       if(value<0.33):
           output2[x][y] = 0
       elif(0.33 < value < 0.66):
           output2[x][y] = 1
       else:
           output2[x][y] = 2
    
    
    return output2
 
predict(Testfeatures1, *parameters)


#nueral network classification by using libraries
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense
#from keras import losses

xTrain, xTest, yTrain, yTest = train_test_split(iris_features1, iris_labels1, test_size = .2, random_state = 5)

model = Sequential()
model.add(Dense(1, input_dim = 4, activation = 'sigmoid'))
model.compile(loss = 'binary_crossentropy', optimizer = 'adam', metric = 'accuracy')
model.fit(xTrain, yTrain, validation = (xTest, yTest), epochs = 100, batchsize = 30)


# reference
# https://github.com/Abhicoder1999/Iris-flower-classifier/blob/master/Iris_classifierV1.0.py
