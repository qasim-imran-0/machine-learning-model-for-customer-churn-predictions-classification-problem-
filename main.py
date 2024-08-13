import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report
import joblib

df = pd.read_csv('Churn_Modelling.csv')
df.drop(['RowNumber', 'CustomerId', 'Surname', 'CreditScore', 'Geography'], axis=1, inplace=True)


label_encoder = LabelEncoder()
df['Gender'] = label_encoder.fit_transform(df['Gender'])  # 0 for Female, 1 for Male


X = df.drop(['Exited'], axis=1)
y = df['Exited']


scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.42, random_state=101)


model = RandomForestClassifier()
model.fit(X_train, y_train)


# prediction = model.predict(X_test)
# print(classification_report(y_test, prediction))
# print('\n')
# print(confusion_matrix(y_test, prediction))


joblib.dump(model, 'churn_predictor.pkl')
joblib.dump(scaler, 'scaler.pkl')
joblib.dump(label_encoder, 'label_encoder.pkl')


import streamlit as st
import pandas as pd
import joblib


model = joblib.load('churn_predictor.pkl')
scaler = joblib.load('scaler.pkl')
label_encoder = joblib.load('label_encoder.pkl')


st.title('Customer Churn Prediction')
st.markdown("Enter the customer details to get a churn prediction.")


gender = st.selectbox("Gender", ["Female", "Male"])
age = st.number_input("Age", min_value=0)
tenure = st.number_input("Tenure", min_value=0)
balance = st.number_input("Balance", min_value=0.0)
num_of_products = st.slider("NumOfProducts", min_value=1, max_value=4)
has_cr_card = st.selectbox("Has Credit Card", ["No", "Yes"])
is_active_member = st.selectbox("Is Active Member", ["No", "Yes"])
estimated_salary = st.number_input("Estimated Salary", min_value=0.0)


gender_encoded = 1 if gender == "Male" else 0
has_cr_card = 1 if has_cr_card == "Yes" else 0
is_active_member = 1 if is_active_member == "Yes" else 0


new_data = pd.DataFrame(
    [[gender_encoded, age, tenure, balance, num_of_products, has_cr_card, is_active_member, estimated_salary]],
    columns=['Gender', 'Age', 'Tenure', 'Balance', 'NumOfProducts', 'HasCrCard', 'IsActiveMember', 'EstimatedSalary']
)


new_data_scaled = scaler.transform(new_data)


prediction = model.predict(new_data_scaled)




if st.button('Predict'):
    st.write("Prediction (0 = No Churn, 1 = Churn):", prediction[0])

