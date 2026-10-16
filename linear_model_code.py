
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model   import LinearRegression
import streamlit as st
st.title("AI Student package  Predictor")
st.write("This is a simple web application that predicts the package of a student based on their CGPA.")
#loading data
data=pd.read_csv("cgpa.csv")
print(data)
#check null
print(data.isnull().sum())
#define X and Y
x=data.iloc[:,[-2]].values
y=data.iloc[:,[-1]].values
print(x.shape,y.shape)
#split the data
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2)
print(x_train,x_test,y_train,y_test)
#define model
lr=LinearRegression()
lr.fit(x_train,y_train)
#test the model
y_pred=lr.predict(x_test)
print(y_pred)
print(y_pred.shape)
#find the accuracy of train and test
trainaccuracy=lr.score(x_train,y_train)
testaccuracy=lr.score(x_test,y_test)
print(trainaccuracy)
print(testaccuracy)
cgpa = st.text_input("ENTER YOUR CGPA")
if st.button("Predict package"):
    try:
        cgpa = float(cgpa)
        if cgpa < 0 or cgpa > 10:
            st.error("Please enter a valid CGPA between 0 to 10.")
        else:
            prediction=lr.predict(np.array([[cgpa]]))
            st.success(f" Predicted Package : {prediction[0][0]:.2f} LPA")
    except ValueError:
        st.error("Please enter a valid number for CGPA.")