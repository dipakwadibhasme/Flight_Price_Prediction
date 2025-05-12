import streamlit as st

# # Set page config
st.set_page_config(
    page_title='Home Page',
    layout='centered', 
    page_icon='✈️'
)

st.title('Flight Price Prediction App')

st.image('image.png')

st.markdown(
    "<p style='text-align: center; font-size:18px;'>This application uses a machine learning " \
    "regression model to predict flight ticket prices based on features such as Date_Of_Journey,airline, source," \
    " destination, departure time, Arrival Time and flight duration.</p>",
    unsafe_allow_html=True
)