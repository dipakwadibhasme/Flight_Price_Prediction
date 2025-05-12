import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

st.set_page_config(page_title='Flight Price Analysis')

df = pd.read_csv('Data_train_cleaned.csv')

def load_airline_details(Airline):
    airline_list = ['IndiGo', 'Air India', 'Jet Airways', 'SpiceJet',
       'Multiple carriers', 'GoAir', 'Vistara', 'Air Asia','Trujet']
    
    filtered_df = df[df['Airline'] == selected_airline]
    st.title(Airline)

    
    # 1. Total Flights
    st.subheader("✈️ Total Flights")
    st.metric(label="Flights Available", value=len(filtered_df))

    # 2. Price Summary
    st.subheader("💰 Price Summary")
    price_stats = filtered_df['Price'].agg(['min', 'max', 'mean']).rename({'min': 'Min Price', 'max': 'Max Price', 'mean': 'Avg Price'})
    st.dataframe(price_stats.to_frame().T.style.format("${:.2f}"))
    
    # 3. Price Distribution
    st.subheader("📊 Price Distribution")
    if 'Price' in filtered_df.columns:
        fig1, ax1 = plt.subplots()
        sns.histplot(filtered_df['Price'], kde=True, ax=ax1, color='teal')
        ax1.set_xlabel("Price")
        ax1.set_ylabel("Number of Flights")
        ax1.set_title(f"{selected_airline} - Price Distribution")
        st.pyplot(fig1)
    
    # 4. Price Boxplot
    st.subheader("📦 Price Spread (Boxplot)")
    fig2, ax2 = plt.subplots()
    sns.boxplot(x=filtered_df['Price'], ax=ax2, color='orange')
    ax2.set_title(f"{selected_airline} - Boxplot")
    st.pyplot(fig2)

# 5. Route Summary (if 'Source' and 'Destination' available)
    if 'Source' in filtered_df.columns and 'Destination' in filtered_df.columns:
        st.subheader("🗺️ Top Routes")
        route_counts = filtered_df.groupby(['Source', 'Destination']).size().reset_index(name='Flights')
        top_routes = route_counts.sort_values(by='Flights', ascending=False).head(5)
        st.dataframe(top_routes)

# 6. Duration Analysis (if available)
    if 'Duration' in filtered_df.columns:
        st.subheader("⏱️ Duration Statistics")
        st.write(filtered_df['Duration'].describe())

# 7. Show Raw Data
    with st.expander("🔍 View All Flight Data for this Airline"):
        st.dataframe(filtered_df)

def load_source_details(Source):
    source_list = ['Banglore', 'Kolkata', 'Delhi', 'Chennai', 'Mumbai']

    filtered_df = df[df['Source'] == selected_source]
    st.title(Source)

    # 1. Total flights
    st.subheader("✈️ Total Flights")
    st.metric(label="Flights Available", value=len(filtered_df))

     # 2. Price Summary
    st.subheader("💰 Price Summary")
    price_summary = filtered_df['Price'].agg(['min', 'max', 'mean']).rename({
        'min': 'Min Price',
        'max': 'Max Price',
        'mean': 'Avg Price'
    })
    st.dataframe(price_summary.to_frame().T.style.format("${:.2f}"))
    
    # 3. Price Distribution Plot
    st.subheader("📊 Price Distribution")
    if 'Price' in filtered_df.columns:
        fig1, ax1 = plt.subplots()
        sns.histplot(filtered_df['Price'], kde=True, ax=ax1, color='green')
        ax1.set_title(f"{selected_source} - Price Distribution")
        ax1.set_xlabel("Price")
        ax1.set_ylabel("Frequency")
        st.pyplot(fig1)

    # 4. Airline Overview
    if 'Airline' in filtered_df.columns:
        st.subheader("🛫 Popular Airlines from This Source")
        airline_counts = filtered_df['Airline'].value_counts().head(5)
        st.bar_chart(airline_counts)

    # 6. Duration Analysis
    if 'Duration' in filtered_df.columns:
        st.subheader("⏱️ Flight Duration Stats")
        st.write(filtered_df['Duration'].describe())


def load_destination_details(selected_destination):
    filtered_df = df[df['Destination'] == selected_destination]

    st.title(f"Flights to {selected_destination}")

    # Show number of flights
    st.write(f"✈️ Total Flights Available to **{selected_destination}**: {len(filtered_df)}")
    
     # Show min, max, mean price
    st.subheader("📊 Price Summary")
    price_summary = filtered_df['Price'].agg(['min', 'max', 'mean']).rename({
        'min': 'Minimum Price',
        'max': 'Maximum Price',
        'mean': 'Average Price'
    })
    st.dataframe(price_summary.to_frame(name='INR'))

    # Show airlines and their average prices
    st.subheader("🛫 Average Price by Airline")
    avg_airline_price = filtered_df.groupby('Airline')['Price'].mean().sort_values()
    st.bar_chart(avg_airline_price)

def date_details(selected_date_of_journey):
    # Convert the 'Date_of_Journey' column to datetime format
    df['Date_of_Journey'] = pd.to_datetime(df['Date_of_Journey'],dayfirst=True, errors='coerce')
    # Extract Month and Year from 'Date_of_Journey'
    df['Month'] = df['Date_of_Journey'].dt.month
    df['Year'] = df['Date_of_Journey'].dt.year

    
    # Filter the data for the selected month
    monthly_data = df[df['Month'] == selected_month]

    st.title(f"Flights of Month {selected_month}")

    
    if monthly_data.empty:
        st.write(f"No journeys found for month {selected_month}.")
    else:
        # Show total number of journeys for the selected month
        total_flights = len(monthly_data)
        st.write(f"Total number of flights operated in month {selected_month}: {total_flights}")

    # Show the flights operated by each airline
        st.subheader(f"🛫 Flights Operated by Airline in Month {selected_month}")
        flights_by_airline = monthly_data.groupby('Airline').size().sort_values(ascending=False)

    # Plot the number of flights operated by each airline
        fig, ax = plt.subplots()
        flights_by_airline.plot(kind='bar', ax=ax, color='skyblue')
        ax.set_xlabel('Airline')  # X-axis label
        ax.set_ylabel('Number of Flights')  # Y-axis label
        ax.set_title(f'Flights Operated by Airline in Month {selected_month}')  # Title
        st.pyplot(fig)  # Display the plot in Streamlit

    # Show the average price for each airline in the selected month
        st.subheader(f"💰 Average Price per Airline in Month {selected_month}")
        avg_price_by_airline = monthly_data.groupby('Airline')['Price'].mean().sort_values(ascending=False)
    
    # Plot the average price of each airline
        fig, ax = plt.subplots()
        avg_price_by_airline.plot(kind='bar', ax=ax, color='lightcoral')
        ax.set_xlabel('Airline')  # X-axis label
        ax.set_ylabel('Average Price (₹)')  # Y-axis label
        ax.set_title(f'Average Price per Airline in Month {selected_month}')  # Title
        st.pyplot(fig)  # Display the plot in Streamlit

def weekwise_analysis_in_month(df, selected_month):
    st.subheader(f"📅 Week-wise Flight Analysis in Month {selected_month}")

     # Convert date column
    df['Date_of_Journey'] = pd.to_datetime(df['Date_of_Journey'], dayfirst=True, errors='coerce')

    # Filter data for the selected month
    monthly_data = df[df['Date_of_Journey'].dt.month == selected_month]

    if monthly_data.empty:
        st.warning("No flight data available for the selected month.")
        return

    # Extract ISO week numbers
    monthly_data['Week_Number'] = monthly_data['Date_of_Journey'].dt.isocalendar().week

    # Show all available weeks in that month
    available_weeks = sorted(monthly_data['Week_Number'].unique())
    selected_week = st.selectbox("Select Week Number in the Month", available_weeks)

    # Filter data for selected week
    week_data = monthly_data[monthly_data['Week_Number'] == selected_week]

    if week_data.empty:
        st.warning(f"No data for Week {selected_week} in Month {selected_month}")
        return

    # Total flights that week
    st.write(f"✈️ Total Flights in Week {selected_week}: {len(week_data)}")

    # Flights per airline
    airline_counts = week_data['Airline'].value_counts()
    st.write("Flights per Airline:")
    st.bar_chart(airline_counts)

    # Average price per airline
    avg_price = week_data.groupby('Airline')['Price'].mean().sort_values(ascending=False)
    st.write("Average Price per Airline:")
    st.bar_chart(avg_price)


def total_stops(selected_stops):
    st.subheader(f"✈️ Analysis for Flights with {selected_stops} Stop(s)")

    # Filter the dataframe based on selected stops
    filtered_df = df[df['Total_Stops'] == selected_stops]

    if filtered_df.empty:
        st.write("No flights found for the selected number of stops.")
        return

    # Show total flights
    total_flights = filtered_df.shape[0]
    st.write(f"📊 Total Flights: {total_flights}")

    # Average price for selected stops
    avg_price = filtered_df['Price'].mean()
    st.write(f"💸 Average Price: ₹{avg_price:,.2f}")

    # Airlines operating with selected stops
    st.subheader("🛫 Airlines with Selected Stops")
    airline_counts = filtered_df['Airline'].value_counts()
    st.bar_chart(airline_counts)

    st.subheader("💸 Average Price by Number of Stops")
    avg_price_by_stops = df.groupby('Total_Stops')['Price'].mean().sort_values()
    st.bar_chart(avg_price_by_stops)
    st.write(avg_price_by_stops)

    st.subheader("🛫 Airlines per Stop Category")
    airline_stop = df.groupby(['Airline', 'Total_Stops']).size().unstack().fillna(0)
    st.dataframe(airline_stop)

    # Source distribution
    st.subheader("🌍 Sources with Selected Stops")
    source_counts = filtered_df['Source'].value_counts()
    st.bar_chart(source_counts)

    # Destination distribution
    st.subheader("🌍 Destinations with Selected Stops")
    dest_counts = filtered_df['Destination'].value_counts()
    st.bar_chart(dest_counts)

def Dep_Time():

    st.title("Departure Time Analysis")

    # Convert 'dep_time' to datetime and extract hour
    df['Dep_Hour'] = df['Dep_Time'].str.split(':').str[0].astype(int)
    df['Dep_Minute'] = df['Dep_Time'].str.split(':').str[1].astype(int)
    # Define time-of-day category
    def get_time_of_day(hour):
        if 5 <= hour < 12:
            return 'Morning'
        elif 12 <= hour < 17:
            return 'Afternoon'
        elif 17 <= hour < 21:
            return 'Evening'
        else:
            return 'Night'

    df['time_of_day'] = df['Dep_Hour'].apply(get_time_of_day)

    # Streamlit filter: select hour range
    st.sidebar.header("Filter Departure Hour")
    selected_hours = st.sidebar.slider("Select hour range:", 0, 23, (6, 18))

    # Filter DataFrame based on selected hour range
    filtered_df = df[(df['Dep_Hour'] >= selected_hours[0]) & (df['Dep_Hour'] <= selected_hours[1])]

    # Plot 1: Number of flights by departure hour
    st.subheader("Number of Flights by Departure Hour")
    hour_counts = filtered_df['Dep_Hour'].value_counts().sort_index()
    st.bar_chart(hour_counts)

    # Plot 2: Number of flights by time of day
    st.subheader("Number of Flights by Time of Day")
    tod_counts = filtered_df['time_of_day'].value_counts()
    st.bar_chart(tod_counts)

    # Show filtered data
    st.subheader("Filtered Data Based on Departure Time")
    st.dataframe(filtered_df)

def Arrival_Time():

    st.title("Arrival Time Analysis")

    # Extract the time part (before space if a date is present)
    df['Time_Only'] = df['Arrival_Time'].str.split(' ').str[0]

    # Extract Hour and Minute
    df['Arrival_Hour'] = df['Time_Only'].str.split(':').str[0].astype(int)
    df['Arrival_Minute'] = df['Time_Only'].str.split(':').str[1].astype(int)


    # Define time-of-day categories
    def get_time_of_day(hour):
        if 5 <= hour < 12:
            return 'Morning'
        elif 12 <= hour < 17:
            return 'Afternoon'
        elif 17 <= hour < 21:
            return 'Evening'
        else:
            return 'Night'

    df['arr_time_of_day'] = df['Arrival_Hour'].apply(get_time_of_day)

    # Streamlit sidebar filter
    st.sidebar.header("Filter Arrival Hour")
    selected_hour_range = st.sidebar.slider("Select arrival hour range", 0, 23, (6, 18))

    # Filter based on selected hour range
    filtered_df = df[(df['Arrival_Hour'] >= selected_hour_range[0]) & (df['Arrival_Hour'] <= selected_hour_range[1])]

    # Plot 1: Arrival counts by hour
    st.subheader("Number of Arrivals by Hour")
    hourly_arrivals = filtered_df['Arrival_Hour'].value_counts().sort_index()
    st.bar_chart(hourly_arrivals)

    # Plot 2: Arrival counts by time of day
    st.subheader("Number of Arrivals by Time of Day")
    time_of_day_counts = filtered_df['arr_time_of_day'].value_counts()
    st.bar_chart(time_of_day_counts)

    # Optional: Show filtered data
    st.subheader("Filtered Arrival Data")
    st.dataframe(filtered_df)

def Duration_Time():

    st.title("Duration Time Analysis")
    
    # Function to extract hours and minutes
    def extract_time(duration):
        hours = 0
        minutes = 0

        if 'h' in duration:
            hours = int(duration.split('h')[0])
        if 'm' in duration:
            minutes = int(duration.split('m')[0].split()[-1])

        return hours, minutes

    # Applying the function
    df[['Duration_hour', 'Duration_Minute']] = df['Duration'].apply(lambda x: pd.Series(extract_time(x)))

    # Total duration in minutes
    df['Total_Duration_Minutes'] = df['Duration_hour'] * 60 + df['Duration_Minute']

    # Sidebar filter
    max_duration = int(df['Total_Duration_Minutes'].max())
    selected_range = st.slider("Select duration (minutes)", 0, max_duration, (0, 600))

    filtered_df = df[
        (df['Total_Duration_Minutes'] >= selected_range[0]) &
        (df['Total_Duration_Minutes'] <= selected_range[1])
    ]

    # Plot: Histogram of durations
    st.subheader("Flight Duration Distribution (in Minutes)")
    fig, ax = plt.subplots()
    sns.histplot(filtered_df['Total_Duration_Minutes'], bins=30, kde=True, ax=ax)
    ax.set_xlabel("Duration (minutes)")
    ax.set_ylabel("Flight Count")
    st.pyplot(fig)

    # Show filtered data
    st.subheader("Filtered Flights")
    st.dataframe(filtered_df)

    st.title("Flight Price vs Duration")

    # Create the scatter plot
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.scatterplot(x='Duration_hour', y='Price', data=df, ax=ax)
    ax.set_title("Flight Price vs Duration")
    ax.set_xlabel("Duration (Hours)")
    ax.set_ylabel("Price")

    # Show in Streamlit
    st.pyplot(fig)

        
st.sidebar.title('Flight price Analysis')

option = st.sidebar.selectbox('select one',['Airline','Source','Destination','Total_Stops','Date_of_Journey',
'Dep_Time','Arrival_Time','Duration'])


if option == 'Airline':
    airline = df['Airline'].unique().tolist()
    selected_airline = st.sidebar.selectbox('Airline',airline)

    if st.sidebar.button('Find Airline Details'):
        load_airline_details(selected_airline)

elif option=='Source':
    source = df['Source'].unique().tolist()
    selected_source = st.sidebar.selectbox('Source',source)
    if st.sidebar.button('Find Source Details'):
        load_source_details(selected_source)

elif option=='Destination':
    destination = df['Destination'].unique().tolist()
    selected_destination = st.sidebar.selectbox('Destination',destination)
    if st.sidebar.button('Find Destination Details'):
        load_destination_details(selected_destination)

elif option == 'Date_of_Journey':
    month = pd.to_datetime(df['Date_of_Journey']).dt.month.unique()
    #selected_month = st.sidebar.selectbox('Select Date of Journey', month)
    selected_month = st.sidebar.selectbox('Select Month for Analysis', range(1, 13))
    if st.sidebar.button('Find Month-wise Details'):
        date_details(selected_month)

    # Button for week-wise analysis within selected month
    if st.sidebar.button('Find Week-wise Details in Month'):
        weekwise_analysis_in_month(df, selected_month)

elif option == 'Total_Stops':
    stops = df['Total_Stops'].unique().tolist()
    selected_stops = st.sidebar.selectbox('Select Number of Stops', stops)

    if st.sidebar.button('Analyze Total Stops'):
        total_stops(selected_stops)

elif option == 'Dep_Time':
    Dep_Time()

elif option == 'Arrival_Time':
    Arrival_Time()

elif option == 'Duration':
    Duration_Time()

