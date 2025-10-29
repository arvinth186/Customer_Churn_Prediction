import pandas as pd
import numpy as np
import streamlit as st
from sklearn.preprocessing import StandardScaler,LabelEncoder,OneHotEncoder
import tensorflow as tf
import pickle


model = tf.keras.models.load_model('model.h5')

with open('label_encoder_gender.pkl','rb') as f:
    label_encoder_gender = pickle.load(f)
with open('one_hot_encoder_geo.pkl','rb') as f:
    one_hot_encoder_geo = pickle.load(f)
with open('scaler.pkl','rb') as f:
    scaler = pickle.load(f)


st.title("Customer Churn Prediction")
st.write("Enter customer details to predict churn probability.")

geography = st.selectbox("Geography",one_hot_encoder_geo.categories_[0])
gender = st.selectbox("Gender",label_encoder_gender.classes_)
age = st.slider("Age",18,92)
balance = st.number_input("Balance",0.0,250000.0,step=100.0)
credit_score = st.number_input("Credit Score")
estimated_salary = st.number_input("Estimated Salary")
tenure = st.slider("Tenure",0,10)
num_of_products = st.slider("Number of Products",1,4)
has_cr_card = st.selectbox("Has Credit Card",[0,1])
is_active_member = st.selectbox("Is Active Member",[0,1])



input_data = pd.DataFrame({
    'CreditScore':[credit_score],
    'Gender':[label_encoder_gender.transform([gender])[0]],
    'Age':[age],
    'Tenure':[tenure],
    'Balance':[balance],
    'NumOfProducts':[num_of_products],
    'HasCrCard':[has_cr_card],
    'IsActiveMember':[is_active_member],
    'EstimatedSalary':[estimated_salary]
})

# One-hot encode Geography
geo_encoded = one_hot_encoder_geo.transform([[geography]]).toarray()
geo_df = pd.DataFrame(geo_encoded,columns=one_hot_encoder_geo.get_feature_names_out(['Geography']))

# Combine encoded geography with other input data
input_data = pd.concat([input_data.reset_index(drop=True),geo_df],axis=1)

# Scale the input data
input_data_scaled = scaler.transform(input_data)

# Predict churn probability
prediction = model.predict(input_data_scaled)
churn_probability = prediction[0][0]

if st.button("Predict Churn Probability"):
    if churn_probability >= 0.5:
        st.error(f"Churn Probability: {churn_probability:.2f} - High risk of churn.")
    else:
        st.success(f"Churn Probability: {churn_probability:.2f} - Low risk of churn.")
