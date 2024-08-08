import streamlit as st
import pandas as pd
import numpy as np
import pickle
import joblib
from sklearn.preprocessing import LabelEncoder

# Load the model
model = joblib.load('news_model_lgb.pkl')

# Load the Vectorizer
vectorizer = joblib.load('vectorizer.pkl')

# Streamlit app
st.set_page_config(page_title="Swahili News Category Classification", page_icon=":bar_chart:", layout="centered", initial_sidebar_state="expanded")

st.title('Swahili News Category Classification')

input_data = {}

st.sidebar.header('Swahili News Classification')
st.sidebar.image('blossom.jpeg')
st.sidebar.divider()
st.sidebar.subheader('Basic Interactions')
st.sidebar.markdown('''
    + Enter your Swahili news article in the text area.
    + Click on the Classify button.
    + View your news classification in the result area below.                                
''')

# Creating input boxes for each feature
input_data['content'] = st.text_area(label='News Contents', height=350, help='Enter your news content here')

# Predict button
if st.button('Classify'):
    try:
        # Transform the input text using the loaded vectorizer
        input_vector = vectorizer.transform([input_data['content']])

        # Convert to dense array and ensure float32 data type
        input_vector_dense = input_vector.toarray().astype(np.float32)

        # Make prediction
        prediction = model.predict(input_vector_dense)
        if prediction[0] == 0.0:
            st.write("Prediction Category: Biashara")
            st.write("Prediction Category: Business")
        elif prediction[0] == 1.0:
            st.write("Prediction Category: Burudani")
            st.write("Prediction Category: Entertainment")
        elif prediction[0] == 2.0:
            st.write("Prediction Category: Kimataifa")
            st.write("Prediction Category: International")
        elif prediction[0] == 3.0:
            st.write("Prediction Category: Kitaifa")
            st.write("Prediction Category: National")
        elif prediction[0] == 4.0:
            st.write("Prediction Category: Michezo")
            st.write("Prediction Category: Sports")
    except Exception as e:
        st.write("Error during prediction:", str(e))
        st.write("Input shape:", input_vector.shape)
        st.write("Input type:", input_vector.dtype)
