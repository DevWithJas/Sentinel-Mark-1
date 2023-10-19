import streamlit as st
import pandas as pd
from streamlit_folium import folium_static
import numpy as np
import time
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from twilio.rest import Client
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from lime.lime_tabular import LimeTabularExplainer
from sklearn.metrics import mean_squared_error
import altair as alt
from altair import Chart
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


# Set the page to full width
st.set_page_config(layout="wide")

# Streamlit app title
st.title("Sentinel Mark 1")

# File upload section
st.sidebar.header("Upload Crime Analysis File")
uploaded_file = st.sidebar.file_uploader("Choose a CSV file", type=["csv"])

# Display data if the file is uploaded
if uploaded_file is not None:
    # Read the uploaded CSV file into a DataFrame
    df = pd.read_csv(uploaded_file)

    # Define keywords to filter crime descriptions
    keywords = ['Child', 'Rape', 'Drugs']

    # Filter the DataFrame to include only rows with crime descriptions containing keywords
    filtered_df = df[df['Crm Cd Desc'].str.contains('|'.join(keywords), case=False, na=False)]


    # Group the data by 'AREA NAME' and 'Crm Cd Desc' and count the number of crimes in each category
    crime_counts_by_area_and_description = filtered_df.groupby(['AREA NAME', 'Crm Cd Desc']).size().unstack(fill_value=0)

    # Plot a bar chart to visualize the number of crimes by crime description in each area
    st.header("Number of Crimes by Crime Description in Each Area")
    plt.figure(figsize=(14, 8))
    crime_counts_by_area_and_description.plot(kind='bar', stacked=True, edgecolor='black', cmap='tab20c')
    plt.xlabel('Area Name')
    plt.ylabel('Number of Crimes')
    plt.title('Number of Crimes by Crime Description in Each Area')
    plt.xticks(rotation=90)  # Rotate x-axis labels for better readability
    plt.tight_layout()
    plt.legend(title='Crime Description', bbox_to_anchor=(1.05, 1), loc='upper left')

    # Show the bar chart
    st.pyplot(plt)


    # Create a list to store training loss values
    training_losses = []

    # Function to predict 3 LAT and LON values with trust levels using a complex deep neural network
    class ComplexCrimeModel(nn.Module):
        def __init__(self):
            super(ComplexCrimeModel, self).__init__()
            self.fc1 = nn.Linear(2, 256)
            self.fc2 = nn.Linear(256, 256)
            self.fc3 = nn.Linear(256, 128)
            self.fc4 = nn.Linear(128, 2)

        def forward(self, x):
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            x = F.relu(self.fc3(x))
            x = self.fc4(x)
            return x

    # Create a PyTorch model
    model = ComplexCrimeModel()

    # Define the loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)

    # Prepare the data for training
    X = filtered_df[['LAT', 'LON']].values
    y = filtered_df[['LAT', 'LON']].values

    # Limit the number of predictions to 3
    num_predictions = 3

    if len(X) >= num_predictions:
        X = X[:num_predictions]
        y = y[:num_predictions]

        # Convert data to PyTorch tensors
        X_tensor = torch.tensor(X, dtype=torch.float32)
        y_tensor = torch.tensor(y, dtype=torch.float32)

        # Training loop
        num_epochs = 100
        batch_size = 64

        for epoch in range(num_epochs):
            model.train()
            optimizer.zero_grad()
            y_pred = model(X_tensor)
            loss = criterion(y_pred, y_tensor)
            loss.backward()
            optimizer.step()

            st.spinner(f"Training Neural Network (Epochs: {num_epochs}, Epoch: {epoch + 1})...")
            
            # Store the loss for dynamic graph
            training_losses.append(loss.item())

        st.spinner("Training Neural Network...Done!")

        # Make predictions (simulating some processing time)
        with st.spinner("Making predictions..."):
            time.sleep(3)  # Simulate processing time

        # Evaluate the model
        model.eval()
        with torch.no_grad():
            X_tensor = torch.tensor(X, dtype=torch.float32)
            predictions = model(X_tensor).numpy()

        # Calculate trust levels based on MAE
        mae = np.mean(np.abs(y - predictions), axis=1)
        trust_levels = 1 - mae / np.linalg.norm(y - predictions, axis=1)

        # Calculate RMSE for LAT and LON
        rmse_lat = np.sqrt(mean_squared_error(y[:, 0], predictions[:, 0]))
        rmse_lon = np.sqrt(mean_squared_error(y[:, 1], predictions[:, 1]))

        # Create a new DataFrame with only the top 3 predictions
        top_predictions_df = pd.DataFrame({'LAT': X[:, 0], 'LON': X[:, 1], 'Trust Level': trust_levels})

        # Print RMSE for LAT and LON
        st.header("Model Accuracy (RMSE)")
        st.write(f"RMSE for LAT: {rmse_lat}")
        st.write(f"RMSE for LON: {rmse_lon}")

        # Create a new DataFrame with only the top 3 predictions
        top_predictions_df = pd.DataFrame({'LAT': X[:, 0], 'LON': X[:, 1], 'Trust Level': trust_levels})

        # Get the top LAT and LON values with the highest trust levels
        top_predictions = top_predictions_df.nlargest(num_predictions, 'Trust Level')

        # Create a map using Folium centered around the top 3 predicted locations
        m = folium.Map(location=[top_predictions['LAT'].mean(), top_predictions['LON'].mean()], zoom_start=15)

        # Add markers for the top trust level locations
        for _, row in top_predictions.iterrows():
            folium.Marker([row['LAT'], row['LON']], tooltip=f"Trust Level: {row['Trust Level']}").add_to(m)

        # Display the map using streamlit-folium, using 100% of the screen width
        folium_static(m)

        # Display the top trust levels along with LAT and LON in a table
        st.header(f"Top {num_predictions} Trust Levels with LAT and LON:")
        st.table(top_predictions[['LAT', 'LON', 'Trust Level']])

        # Include ARIMA and SARIMA outlier detection with scatter plots for LAT and LON
        st.header("ARIMA & SARIMA Outlier Detection")
        st.write("This scatter plot shows ARIMA and SARIMA outlier detection for both LAT and LON.")

        # Perform ARIMA outlier detection on the 'LAT' column
        lat_data = filtered_df['LAT'].values
        model_lat = ARIMA(lat_data, order=(5, 1, 0))
        model_fit_lat = model_lat.fit()
        residuals_lat = model_fit_lat.resid
        outlier_threshold_lat = 2.0
        outliers_lat = np.where(np.abs(residuals_lat) > outlier_threshold_lat)

        # Perform SARIMA outlier detection on the 'LAT' column
        sarima_model_lat = SARIMAX(lat_data, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12))
        sarima_model_lat_fit = sarima_model_lat.fit(disp=False)
        sarima_residuals_lat = sarima_model_lat_fit.resid
        sarima_outliers_lat = np.where(np.abs(sarima_residuals_lat) > outlier_threshold_lat)


        # Perform ARIMA outlier detection on the 'LON' column
        lon_data = filtered_df['LON'].values
        model_lon = ARIMA(lon_data, order=(5, 1, 0))
        model_fit_lon = model_lon.fit()
        residuals_lon = model_fit_lon.resid
        outlier_threshold_lon = 2.0
        outliers_lon = np.where(np.abs(residuals_lon) > outlier_threshold_lon)

        # Perform SARIMA outlier detection on the 'LON' column
        sarima_model_lon = SARIMAX(lon_data, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12))
        sarima_model_lon_fit = sarima_model_lon.fit(disp=False)
        sarima_residuals_lon = sarima_model_lon_fit.resid
        sarima_outliers_lon = np.where(np.abs(sarima_residuals_lon) > outlier_threshold_lon)

        # Scatter plot for 'LAT' column with ARIMA and SARIMA outliers
        plt.figure(figsize=(10, 6))
        plt.scatter(range(len(lat_data)), lat_data, label='LAT Data')
        plt.scatter(outliers_lat, lat_data[outliers_lat], color='red', marker='o', label='ARIMA Outliers (LAT)')
        plt.scatter(sarima_outliers_lat, lat_data[sarima_outliers_lat], color='blue', marker='x', label='SARIMA Outliers (LAT)')
        plt.xlabel('Data Point')
        plt.ylabel('LAT')
        plt.title('LAT Outliers Detected by ARIMA & SARIMA')
        plt.legend()
        st.pyplot(plt)

        # Scatter plot for 'LON' column with ARIMA and SARIMA outliers
        plt.figure(figsize=(10, 6))
        plt.scatter(range(len(lon_data)), lon_data, label='LON Data')
        plt.scatter(outliers_lon, lon_data[outliers_lon], color='red', marker='o', label='ARIMA Outliers (LON)')
        plt.scatter(sarima_outliers_lon, lon_data[sarima_outliers_lon], color='blue', marker='x', label='SARIMA Outliers (LON)')
        plt.xlabel('Data Point')
        plt.ylabel('LON')
        plt.title('LON Outliers Detected by ARIMA & SARIMA')
        plt.legend()
        st.pyplot(plt)

        #ain, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    

    # Model Training Progress with Dynamic Graph
    st.header("Model Training Progress")
    chart = Chart(pd.DataFrame({'Epoch': range(1, num_epochs + 1), 'Loss': training_losses})).mark_line().encode(
        x='Epoch',
        y='Loss'
    ).interactive()
    st.altair_chart(chart, use_container_width=True)


# Twilio SMS Sender section
st.sidebar.header("Twilio SMS Sender")
twilio_account_sid = st.sidebar.text_input("Twilio Account SID", "")
twilio_auth_token = st.sidebar.text_input("Twilio Auth Token", "")
twilio_phone_number = st.sidebar.text_input("Twilio Phone Number", "")
recipient_phone_number = st.sidebar.text_input("Recipient Phone Number", "")

# Create a button to send the SMS
if st.sidebar.button("Send SMS"):
    # Initialize the Twilio client
    client = Client(twilio_account_sid, twilio_auth_token)

    # Prepare the message with the predicted LAT and LON values
    message = "Hi sir, these are the next hotspots:\n"
    for i, (lat, lon) in enumerate(zip(top_predictions['LAT'], top_predictions['LON']), start=1):
        message += f"{i}. LAT: {lat}, LON: {lon}\n"

    # Create a spinning indicator
    with st.spinner("Sending SMS..."):
        try:
            # Send the SMS message in a single request
            client.messages.create(
                body=message,
                from_=twilio_phone_number,
                to=recipient_phone_number
            )
            st.sidebar.success("SMS sent successfully!")
        except Exception as e:
            st.sidebar.error(f"Error sending SMS: {e}")
