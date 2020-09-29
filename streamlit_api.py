# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

#from flask import Flask,request
import pandas as pd
import numpy as np
import joblib
#import flasgger
#from flasgger import Swagger
import streamlit as st
import os

from PIL import Image

#app=Flask(__name__)

#Swagger(app)

classifier=joblib.load('my_model.pkl')


#@app.route('/')
def welcome():
    return "Welcome All"

#@app.route('/predict',methods=["Get"])
def predict_note_authentication(variance,skewness,curtosis,entropy):
    
    """Let's Authenticate the Banks Note 
    This is using docstrings for specifications.
    ---
    parameters:  
      - name: variance
        in: query
        type: number
        required: true
      - name: skewness
        in: query
        type: number
        required: true
      - name: curtosis
        in: query
        type: number
        required: true
      - name: entropy
        in: query
        type: number
        required: true
    responses:
        200:
            description: The output values
        
    """
    #variance=request.args.get("variance")
    #skewness=request.args.get("skewness")
    #curtosis=request.args.get("curtosis")
    #entropy=request.args.get("entropy")
    prediction=classifier.predict([[variance,skewness,curtosis,entropy]])
    print(prediction)
    return prediction



def main():
    st.title("Bank Notes Authenticator")
    html_temp= """
    <div style= "background-color:tomato;padding:10px">
    <h2 style = "color:white;text-align:center;">Streamlit Bank Authenticator ML App </h2>
    </div>
    """
    st.markdown(html_temp,unsafe_allow_html=True)
    variance= st.text_input("Variance","Type here")
    skewness= st.text_input("Skewness","Type here")
    curtosis= st.text_input("Curtosis","Type here")
    entropy= st.text_input("Entropy","Type here")
    result=""
    if st.button("Predict"):
        result= predict_note_authentication(variance,skewness,curtosis,entropy)
    st.success("The output is {}".format(result))
    if st.button("About"):
        st.text('Lets Learn')
        st.text("Interface built with Streamlit")

if __name__=='__main__':
    main()
