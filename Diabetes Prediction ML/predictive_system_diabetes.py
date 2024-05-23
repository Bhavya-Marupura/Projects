# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np
import pickle

#loading the saved model
#read the file in binary format and load it 
loaded_model = pickle.load(open("C:/Users\Bhavya/Desktop/Bhavya/Sem 5/ML/Project/trained_model.sav",'rb'))

#Predicting using User Input
input_data=[4,110,92,0,0,37.6,0.191,30]
input_data=np.asarray(input_data)
#reshape the array as we are predicting one instance
input_data=input_data.reshape(1,-1)
print('Input Data')
print(input_data)
prediction=loaded_model.predict(input_data)
print(prediction)
if prediction[0]==0:
    print('Person is Not-Diabetic')
else:
    print('Person is Diabetic')