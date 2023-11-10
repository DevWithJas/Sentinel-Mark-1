# Sentinel-Mark-1

The Streamlit app is designed for crime analysis and prediction. It includes features for data upload, crime visualization, neural network training, model accuracy evaluation, and ARIMA/SARIMA outlier detection. The app dynamically displays a welcome message, a progress chart, and a progress bar during processing.

The crime analysis section filters data based on keywords, visualizes crime counts, and utilizes neural networks for location predictions. The model's accuracy is displayed with Root Mean Squared Error (RMSE). The top trust level predictions, along with their locations, are presented on a map.

The app also incorporates ARIMA and SARIMA outlier detection, showcasing scatter plots with detected outliers for both latitude (LAT) and longitude (LON). Additionally, the training progress of the neural network is dynamically visualized.

For added functionality, the app includes a Twilio SMS sender, allowing users to send messages with predicted crime hotspots to specified phone numbers. The code uses various Python libraries, such as Streamlit, Pandas, PyTorch, Altair, Folium, and Twilio, creating a comprehensive tool for crime analysis and prediction.

Dynamic Welcome Message: The app starts with a dynamic welcome message using Markdown, creating an engaging introduction.

Progress Chart and Bar: A temporary loading message is displayed alongside a progress chart and bar, simulating progress with random data. This provides visual feedback to users while the app processes information.

Crime Data Upload: Users can upload crime analysis data in CSV format through the sidebar. The app reads the uploaded data into a DataFrame for further analysis.

Crime Filtering: Crime data is filtered based on keywords, allowing users to focus on specific crime descriptions such as 'Child,' 'Rape,' and 'Drugs.'

Crime Visualization: The app generates a bar chart visualizing the number of crimes by description in each area. This provides a clear overview of crime distribution.

Neural Network Training: A complex deep neural network is employed to predict latitude (LAT) and longitude (LON) values. The app dynamically displays the training progress with a spinning indicator.

Model Accuracy Evaluation: The accuracy of the model is evaluated, and Root Mean Squared Error (RMSE) for LAT and LON is presented to assess the model's performance.

Top Predictions and Map Display: The top predictions with the highest trust levels are displayed on a Folium map. The app dynamically updates the map with markers for each prediction.

ARIMA/SARIMA Outlier Detection: Outliers in LAT and LON are detected using both ARIMA and SARIMA models. Scatter plots visualize the detected outliers.

Model Training Progress Chart: A dynamic chart using Altair shows the training progress of the neural network, updating in real-time.

Twilio SMS Sender: Users can send SMS messages containing predicted crime hotspots using Twilio. The app includes input fields for Twilio credentials and phone numbers.

Comprehensive Description: The entire app's functionality is described, emphasizing the use of Python libraries such as Streamlit, Pandas, PyTorch, Altair, Folium, and Twilio.
