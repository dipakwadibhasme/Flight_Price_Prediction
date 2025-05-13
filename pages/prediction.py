import streamlit as st
import pandas as pd
import numpy as np
import pickle
import datetime

st.set_page_config(page_title='Flight price precictin App')

df = pd.read_csv('df1.csv') 

with open('E:\Streamlit\Flight prediction\pipeline.pkl','rb') as file:
    pipeline = pickle.load(file)

st.header('Enter your Inputs :')

# Airline
Airline = st.selectbox('Airline',df['Airline'].unique().tolist())
# Source
Source = st.selectbox('Source',df['Source'].unique().tolist())
# Destination
Destination = st.selectbox('Destination',df['Destination'].unique().tolist())
# Total_Stops
Total_Stops = st.selectbox('Total_Stops',df['Total_Stops'].unique().tolist())
# Date_of_Journey 
Journey_Date = st.date_input('Date_of_Journey',datetime.date.today())
# Extract Day and Month from Journey_Date
Journey_Day = Journey_Date.day
Journey_Month = Journey_Date.month
# departure time
Dep_Time = st.time_input('Departure Time (24-Hour)', value=datetime.time(23, 15))

# --- Extract Hours and Minutes from Departure Time ---
Dep_hour = Dep_Time.hour
Dep_Minute = Dep_Time.minute
#Arrival Time
Arrival_Time = st.time_input('Arrival Time (24-Hour)', value=datetime.time(1, 30))
# --- Extract Hours and Minutes from Arrival Time ---
Arrival_hour = Arrival_Time.hour
Arrival_Minute = Arrival_Time.minute
# --- Duration Input ---
Duration_hour = st.number_input('Duration Hours', min_value=0, max_value=24, value=1)
Duration_Minute = st.number_input('Duration Minutes', min_value=0, max_value=59, value=30)


if st.button('predict'):
    # form a dataframe
    data = [[Airline,Source,Destination,Total_Stops,Journey_Day,Journey_Month,Dep_hour,Dep_Minute,Arrival_hour,
             Arrival_Minute,Duration_hour,Duration_Minute]]
    columns = ['Airline','Source','Destination','Total_Stops', 'Journey_Day', 'Journey_Month',
                'Dep_hour','Dep_Minute','Arrival_hour','Arrival_Minute','Duration_hour','Duration_Minute']

    # convert to dataframe
    one_df = pd.DataFrame(data,columns=columns)
    #st.dataframe(one_df)
    # predict
    st.text(np.expm1(pipeline.predict(one_df)))
    # display
