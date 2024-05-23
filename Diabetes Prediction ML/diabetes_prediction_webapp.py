# -*- coding: utf-8 -*-
"""
Created on Sun Nov 19 14:09:19 2023

@author: Bhavya
"""

import numpy as np
import pickle
import streamlit as st

#loading the saved model
#read the file in binary format and load it 
loaded_model = pickle.load(open("C:/Users\Bhavya/Desktop/Bhavya/Sem 5/ML/Project/trained_model.sav",'rb'))

#creating a function for prediction
def diabetes_pred(input_data):

    input_data=np.asarray(input_data)
    #reshape the array as we are predicting one instance
    input_data=input_data.reshape(1,-1)
    print('Input Data')
    print(input_data)
    #predicting the output
    prediction=loaded_model.predict(input_data)
    print(prediction)
    if prediction[0]==0:
        return 'Person is Not-Diabetic'
    else:
        return 'Person is Diabetic'

def main():
    
    #Giving a title for our WebApp
    st.title('Diabetes Prediction WebApp')
    
    #taking the input parameters from the user
    Pregnancies = st.text_input('Number of Pregnancies')
    Glucose = st.text_input('Glucose Level')
    BloodPressure = st.text_input('Blood Pressure Value')
    SkinThickness = st.text_input('Skin Thickness Value')
    Insulin = st.text_input('Insulin Level')
    BMI = st.text_input('Body Mass Index (BMI) Value')
    DiabetesPedigreeFunction = st.text_input('Diabetes Pedigree Function Value')
    Age = st.text_input('Age of the User')
    
    #Code for prediction
    diagnosis = " "
    
    if st.button("Diabetes Test Result"):
        diagnosis= diabetes_pred([Pregnancies,Glucose,BloodPressure,SkinThickness,Insulin,BMI,DiabetesPedigreeFunction,Age ])
        
    st.success(diagnosis)
    
    
    
    
    
    
if __name__ == '__main__':
    main()
        
        
        
        
        
        
        