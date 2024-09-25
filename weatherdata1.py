import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

def load_data():
    file_path = "C:\\Users\\murad\\OneDrive\\Documents\\Desktop\\Code projects\\SolarCarApp\\Jan2022.csv"


    data = pd.read_csv(file_path)

    #drop irrelevant or unnecessary columns: especially columns that can't be quantified
    columns_to_drop = ["Date/Time (LST)", "Time (LST)","Longitude (x)", "Latitude (y)", "Station Name", "Climate ID", "Precip. Amount (mm)", 
                    "Temp Flag", "Dew Point Temp Flag", "Rel Hum Flag", 
                    "Precip. Amount Flag", "Wind Dir Flag", "Wind Spd Flag", 
                    "Visibility Flag", "Stn Press Flag", "Hmdx Flag", "Hmdx", "Wind Chill Flag", "Wind Chill", "Weather"]

    data_cleaned = data.drop(columns=columns_to_drop)

    #replace empty values (NaN)
    data_cleaned.replace([np.inf, -np.inf], np.nan, inplace=True)
    data_cleaned = data_cleaned.dropna()
    data_cleaned = data_cleaned.fillna(data_cleaned.mean())




    #select relevant columns for training
    features_columns = ["Temp (°C)", "Dew Point Temp (°C)", "Rel Hum (%)", 
                        "Wind Dir (10s deg)", "Wind Spd (km/h)", "Visibility (km)", "Stn Press (kPa)"]

    #looking back 24 hrs into past
    window_size = 24

    #initialize lists to store data and desired prediction
    X = []
    y = []

    #loop over the data to create sliding windows
    for i in range(len(data_cleaned) - window_size):
        
        #look back 24 hours
        X.append(data_cleaned[features_columns].iloc[i:i + window_size].values)
        
        #save temp. of next hr
        y.append(data_cleaned["Temp (°C)"].iloc[i + window_size])


    X = np.array(X)
    y = np.array(y)

    # Reshape X to 2D: (number_of_samples * time_steps, number_of_features)
    X_reshaped = X.reshape(-1, X.shape[-1])  # X.shape[-1] is the number of features

    #data needs to be scaled ***CHAT GPT USED HERE
    scaler = StandardScaler()

    # Scale the reshaped data
    X_scaled = scaler.fit_transform(X_reshaped)

    # Reshape back to 3D: (number_of_samples, time_steps, number_of_features)
    X_scaled_reshaped = X_scaled.reshape(X.shape)

    np.save('X_data.npy', X)
    np.save('y_data.npy', y)

