import pandas as pd
import streamlit as st
st.header("Salary Prediction")
st.sidebar.header("User Input")
experience=st.sidebar.slider("No. of experience",0,15,4)
cgpa=st.sidebar.slider("Cgpa Score",0,10,5)
age=st.sidebar.slider("Age",18,60,20)
interview_score=st.sidebar.slider("Interview Score",0,100,50)




df=pd.read_csv("salary_predict_dataset.csv")
x=df.iloc[:,:4]
y=df.iloc[:,4:]

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)

from sklearn.linear_model import LinearRegression
lr=LinearRegression()
lr.fit(x_train,y_train)
y_pred=lr.predict([[experience,cgpa,age,interview_score]])
st.subheader("Predicted Salary")
st.write(y_pred)
from sklearn.metrics import r2_score
y_pred1=lr.predict(x_test)
r2=r2_score(y_test,y_pred1)
st.subheader("R2 Score")
st.write(r2)