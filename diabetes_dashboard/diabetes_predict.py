import pandas as pd
import numpy as np
import seaborn as sns
import streamlit as st
import matplotlib.pyplot as plt
import plotly.graph_objs as go
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from PIL import Image


df = pd.read_csv('C:/Users/LaNaturess/Desktop/diabetes_dashboard/source/diabetes_2021_health_indicators_BRFSS2021.csv')
selected_columns = ['Diabetes', 'Race/Ethnicity', 'Physical Activity', 'High BP', 'Gender', 'Age', 'Overweight/Obese', 'Income', 'Smoker']
df_selected = df[selected_columns]


# Convert categorical features to numerical features
df_selected['Gender'] = np.where(df_selected['Gender'] == 'Male', 1, 0)
df_selected['Race/Ethnicity'] = pd.factorize(df_selected['Race/Ethnicity'])[0]

# Handle missing values
df_selected = df_selected.dropna()

# Split the data into training and testing sets
X = df_selected.drop('Diabetes', axis=1)
y = df_selected['Diabetes']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a logistic regression model
model = LogisticRegression(random_state=42)
model.fit(X_train, y_train)

# Evaluate the model's accuracy on the testing data
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

# Create the streamlit dashboard
# Set the page title
st.set_page_config(page_title='Diabetes Prediction Dashboard', page_icon=':pill:', layout='wide')

st.title('Diabetes Prediction Dashboard')


st.subheader('Training Data Stats')
st.write(df.describe())


# Convert the slider and selectbox inputs to the format used by the model
# Define the sidebar
st.sidebar.title('Diabetes Prediction Model')
st.sidebar.write('Please select the values for the following features:')

race_ethnicity = st.sidebar.selectbox('Race/Ethnicity', ['White', 'Black', 'Asian', 'American Indian/Alaskan Native', 'Hispanic', 'Other'])
race_ethnicity_dict = {'White': 1, 'Black': 2, 'Asian': 3, 'American Indian/Alaskan Native': 4, 'Hispanic': 5, 'Other': 6}
race_ethnicity = race_ethnicity_dict[race_ethnicity]

physical_activity = st.sidebar.selectbox('Physical Activity', ['Yes', 'No'])
physical_activity_dict = {'Yes': 1, 'No': 2}
physical_activity = physical_activity_dict[physical_activity]

high_BP = st.sidebar.selectbox('High BP', ['No', 'Yes'])
high_BP_dict = {'No': 1, 'Yes': 2}
high_BP = high_BP_dict[high_BP]

gender = st.sidebar.selectbox('Gender', ['Male', 'Female'])
gender_dict = {'Male': 1, 'Female': 2}
gender = gender_dict[gender]

age = st.sidebar.selectbox('Age', ['Age 18 to 24', 'Age 25 to 34', 'Age 35 to 44', 'Age 45 to 54', 'Age 55 to 64', 'Age 65 or older'])
age_dict = {'Age 18 to 24': 1, 'Age 25 to 34': 2, 'Age 35 to 44': 3, 'Age 45 to 54': 4, 'Age 55 to 64': 5, 'Age 65 or older': 6}
age = age_dict[age]

overweight_obese = st.sidebar.selectbox('Overweight/Obese', ['No', 'Yes'])
overweight_obese_dict = {'No': 1, 'Yes': 2}
overweight_obese = overweight_obese_dict[overweight_obese]


income = st.sidebar.selectbox('Income', ['Less than $15,000', '$15,000 to < $25,000', '$25,000 to < $35,000', '$35,000 to < $50,000', '$50,000 to < $100,000', '$100,000 to < $200,000', '$200,000 or more'])
income_dict = {'Less than $15,000': 1, '$15,000 to < $25,000': 2, '$25,000 to < $35,000': 3, '$35,000 to < $50,000': 4, '$50,000 to < $100,000': 5, '$100,000 to < $200,000': 6, '$200,000 or more': 7}
income = income_dict[income]

smoker = st.sidebar.selectbox('Smoker', ['Current smoker -now smokes every day', 'Current smoker -now smokes some days', 'Former smoker', 'Never smoked'])
smoker_dict = {'Current smoker -now smokes every day': 1, 'Current smoker -now smokes some days': 2, 'Former smoker': 3, 'Never smoked': 4}
smoker = smoker_dict[smoker]



# Create a feature vector
features = [race_ethnicity, physical_activity, high_BP, gender, age, overweight_obese, income, smoker]


# Make a prediction using the model
prediction = model.predict([[race_ethnicity, physical_activity, high_BP, gender, age, overweight_obese, income, smoker]])[0]

# Create a function to make predictions
def predict_diabetes(features):
    prediction = model.predict([features])
    if prediction == 0:
        return 'No Diabetes'
    else:
        return 'Diabetes'


# Create a submit button
submitted = st.sidebar.button('Submit')

# Make a prediction when the button is clicked
if submitted:
    features = [race_ethnicity, physical_activity, high_BP, gender, age, overweight_obese, income, smoker]
    prediction = predict_diabetes(features)


#----------------------------------------------------------------------------------------------------------------------------------

#Show all graphs from jupyter notebook for more detailed visualizations

# Set a title
st.title('Visualizations')


# Load image of graph
diabetes_by_weight_image = Image.open(r'C:\Users\LaNaturess\Desktop\diabetes_dashboard\images\diabetesbyweight.png')

# Display image in Streamlit app
st.image(diabetes_by_weight_image, caption='Diabetes by Weight', width=800)


# Load image of graph
diabetes_by_age_image = Image.open(r'C:\Users\LaNaturess\Desktop\diabetes_dashboard\images\diabetesbyage.png')

# Display image in Streamlit app
st.image(diabetes_by_age_image, caption='Diabetes by Age', width=800)


# Load image of graph
diabetes_by_bp_image = Image.open(r'C:\Users\LaNaturess\Desktop\diabetes_dashboard\images\diabetesbybp.png')

# Display image in Streamlit app
st.image(diabetes_by_bp_image, caption='Diabetes by Blood Pressure', width=800)


# Load image of graph
diabetes_by_gender_image = Image.open(r'C:\Users\LaNaturess\Desktop\diabetes_dashboard\images\diabetesbygender.png')

# Display image in Streamlit app
st.image(diabetes_by_gender_image, caption='Diabetes by Gender', width=800)


# Load image of graph
diabetes_by_heartD_image = Image.open(r'C:\Users\LaNaturess\Desktop\diabetes_dashboard\images\diabetesbyheartdisease.png')

# Display image in Streamlit app
st.image(diabetes_by_heartD_image, caption='Diabetes by Heart Disease', width=800)


# Load image of graph
diabetes_by_smoker_image = Image.open(r'C:\Users\LaNaturess\Desktop\diabetes_dashboard\images\diabetesbysmoker.png')

# Display image in Streamlit app
st.image(diabetes_by_smoker_image, caption='Diabetes by Smoker', width=800)


# Load image of graph
features_importance_image = Image.open(r'C:\Users\LaNaturess\Desktop\diabetes_dashboard\images\featuresimportance.png')

# Display image in Streamlit app
st.image(features_importance_image, caption='Features Importance', width=800)


# Load image of graph
correlation_matrix_image = Image.open(r'C:\Users\LaNaturess\Desktop\diabetes_dashboard\images\correlationmatrix.png')

# Display image in Streamlit app
st.image(correlation_matrix_image, caption='Correlation Matrix', width=1200)


#create a hypeelink to tableau for more visualization

st.subheader('link to the Tableau')
url = "https://public.tableau.com/app/profile/mojtaba.zadaskar/viz/shared/3NPDMTQMQ"
link = f'<a href="{url}" target="_blank">Visualizations with Tableau</a>'
st.markdown(link, unsafe_allow_html=True)



# Show the prediction result
st.subheader('Prediction Result')
st.write(f'The predicted result is: {prediction}')
st.write('Accuracy:', accuracy)

# Check if output result is diabetes
if prediction == "Diabetes":
    # Display recommendations section
    st.header('Recommendations')
    st.write('Here are some recommendations for managing diabetes:')
    st.write('- Maintain a healthy diet and exercise regularly.')
    st.write('- Monitor your blood sugar levels regularly.')
    st.write('- Take your medication as prescribed by your doctor.')


# Check if output result is diabetes
if prediction == "No Diabetes":
    # Display recommendations section
    st.subheader('Keep up the good work')
    st.write('Your Healthy as ever')
   
