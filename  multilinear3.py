import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

# Column names
column_names = [
    "MedInc", "HouseAge", "AveRooms", "AveBedrms",
    "Population", "AveOccup", "Latitude", "Longitude", "MedHouseVal"
]

# Load dataset
df = pd.read_csv("/Users/cex/Desktop/CaliforniaHousing/cal_housing.data", header=None, names=column_names)

# Select all features (excluding the target)
X = df.drop(columns=["MedHouseVal"])
y = df["MedHouseVal"]

# Train model
model = LinearRegression()
model.fit(X, y)

print("Intercept (β₀):", model.intercept_)
print("Coefficients (β):", model.coef_)

# --- Predict from user input ---
try:
    med_inc = float(input("Enter Median Income: "))
    age = float(input("Enter House Age: "))
    avg_rooms = float(input("Enter Average Number of Rooms: "))
    avg_bedrooms = float(input("Enter Average Number of Bedrooms: "))
    population = float(input("Enter Population: "))
    avg_occup = float(input("Enter Average Occupancy: "))
    latitude = float(input("Enter Latitude: "))
    longitude = float(input("Enter Longitude: "))

    input_data = [[
        med_inc, age, avg_rooms, avg_bedrooms,
        population, avg_occup, latitude, longitude
    ]]

    input_df = pd.DataFrame(input_data, columns=X.columns)
    prediction = model.predict(input_df)
    print(f"\n Predicted Median House Value: ${prediction[0]:,.2f}")

except ValueError:
    print(" Invalid input. Please enter numeric values.")


