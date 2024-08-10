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
st.sidebar.image('blossom2.jpeg')
st.sidebar.divider()
st.sidebar.subheader('Basic Interactions')
st.sidebar.markdown('''
    + Enter your Swahili news article in the text area.
    + Click on the Classify button.
    + View your news classification in the result area below.                                
''')
st.sidebar.divider()
st.sidebar.subheader('Classification Boundaries')
st.sidebar.markdown('''
    The model only classifies in 5 categories:
    + National
    + International
    + Sports
    + Business
    + Entertainment                                
''')
st.sidebar.divider()
st.sidebar.subheader('Project Contributors')
st.sidebar.markdown('''
    + [Adjoa Panyin Kuwornu](https://www.linkedin.com/in/adjoapanyinkuwornu/)
    + [Abigail Sapong](https://www.linkedin.com/in/abigail-sapong/)
    + [Simon Odjam](https://www.linkedin.com/in/simon-n-odjam-a50a20178/)
    + [Samuel Efosa-Austin](https://www.linkedin.com/in/samuelobeghe/)
    + [Chigozie Paschal Okafor](http://linkedin.com/in/chigozie-paschal-okafor-itil-346967122)
    + [Japhthah Boateng](https://www.linkedin.com/in/jephthah-boateng/)          
    + [Hemeedah Ibrahim Lawal](https://www.linkedin.com/in/hameedah-lawal-61088510b)
    + [Enoch Boateng](https://www.linkedin.com/in/enoch-boateng)
    + [Ezinne Okoro](https://www.linkedin.com/in/okoro-ezinne-15068728b/)
    + [Ahmad Bilesanmi](https://www.linkedin.com/in/bilesanmi-olorunfemi/)                             
''')

# Creating input boxes for each feature
input_data['content'] = st.text_area(label='News Contents', height=350, help='Enter your news content here')
news_classes = {
    0.0: ["Biashara", "Business"],
    1.0: ["Burudani", "Entertainment"],
    2.0: ["Kimataifa", "International"],
    3.0: ["Kitaifa", "National"],
    4.0: ["Michezo", "Sports"]
}

# Predict button
if st.button('Classify'):
    try:
        if input_data['content'] == '':
            st.write("Classification cannot run for blank text")
        else:
            # Transform the input text using the loaded vectorizer
            input_vector = vectorizer.transform([input_data['content']])

            # Convert to dense array and ensure float32 data type
            input_vector_dense = input_vector.toarray().astype(np.float32)

            # Make prediction
            prediction = model.predict(input_vector_dense)
            st.write("News Article Category (Swahili): " + news_classes[prediction[0]][0])
            st.write("News Article Category: " + news_classes[prediction[0]][1])
    except Exception as e:
        st.write("Error during prediction:", str(e))
        st.write("Input shape:", input_vector.shape)
        st.write("Input type:", input_vector.dtype)
