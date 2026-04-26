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

# Select TWO features and the target
X = df[["AveRooms", "HouseAge"]]
y = df["MedHouseVal"]

# Train model
model = LinearRegression()
model.fit(X, y)

print("Intercept (β₀):", model.intercept_)
print("Coefficients (β):", model.coef_)

# 3D Scatter Plot with Regression Plane
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')

# Scatter actual data
ax.scatter(X["AveRooms"], X["HouseAge"], y, alpha=0.3, label="Data points")

# Create a meshgrid for the plane
x_surf, y_surf = np.meshgrid(
    np.linspace(X["AveRooms"].min(), X["AveRooms"].max(), 50),
    np.linspace(X["HouseAge"].min(), X["HouseAge"].max(), 50)
)
z_surf = (
    model.intercept_
    + model.coef_[0] * x_surf
    + model.coef_[1] * y_surf
)

# Plot regression plane
ax.plot_surface(x_surf, y_surf, z_surf, color='red', alpha=0.5)

# Labels
ax.set_xlabel("AveRooms")
ax.set_ylabel("HouseAge")
ax.set_zlabel("MedHouseVal")
ax.set_title("3D Regression: AveRooms & HouseAge vs Median House Value")

plt.show()

# --- Predict from user input ---