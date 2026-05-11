# Multiple Linear Regression using Diabetes Dataset

# Import libraries
from sklearn.datasets import load_diabetes
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# ---------------------------------------------------
# Load Real-World Dataset
# ---------------------------------------------------

# Diabetes dataset from scikit-learn
diabetes = load_diabetes(as_frame=True)

# Convert dataset into dataframe
df = diabetes.frame

print("First 5 Rows of Dataset:")
print(df.head())

# ---------------------------------------------------
# PART A
# Train model using ONLY TWO FEATURES
# ---------------------------------------------------

# Select two features
X2 = df[['bmi', 'bp']]

# Target variable
y = df['target']

# Split dataset into training and testing sets
X2_train, X2_test, y_train, y_test = train_test_split(
    X2, y, test_size=0.2, random_state=42
)

# Create regression model
model2 = LinearRegression()

# Train model
model2.fit(X2_train, y_train)

# Beta values
print("\nPART A")
print("Intercept (β0):", model2.intercept_)
print("Coefficients (β1, β2):", model2.coef_)

# Predict output
sample_input_2 = [[0.05, 0.03]]

prediction_2 = model2.predict(sample_input_2)

print("\nPrediction using two features:")
print("Input:", sample_input_2)
print("Predicted Output:", prediction_2)



fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')

# Scatter plot
ax.scatter(
    X2['bmi'],
    X2['bp'],
    y
)

# Create regression plane
bmi_range = np.linspace(X2['bmi'].min(), X2['bmi'].max(), 20)
bp_range = np.linspace(X2['bp'].min(), X2['bp'].max(), 20)

bmi_grid, bp_grid = np.meshgrid(bmi_range, bp_range)

target_grid = (
    model2.intercept_
    + model2.coef_[0] * bmi_grid
    + model2.coef_[1] * bp_grid
)

# Plot regression plane
ax.plot_surface(bmi_grid, bp_grid, target_grid, alpha=0.5)

# Labels
ax.set_xlabel("BMI")
ax.set_ylabel("Blood Pressure")
ax.set_zlabel("Disease Progression")

ax.set_title("Multiple Linear Regression")

plt.show()


# Select multiple features
X_multi = df[['bmi', 'bp', 's1', 's2', 's5']]

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X_multi, y, test_size=0.2, random_state=42
)

# Create model
model_multi = LinearRegression()

# Train model
model_multi.fit(X_train, y_train)

# Print beta values
print("\nPART B")

print("Intercept (β0):", model_multi.intercept_)

print("\nCoefficients:")
print("β1 (bmi) =", model_multi.coef_[0])
print("β2 (bp) =", model_multi.coef_[1])
print("β3 (s1) =", model_multi.coef_[2])
print("β4 (s2) =", model_multi.coef_[3])
print("β5 (s5) =", model_multi.coef_[4])

# Predict output
sample_input_multi = [[0.05, 0.03, -0.02, -0.01, 0.04]]

prediction_multi = model_multi.predict(sample_input_multi)

print("\nPrediction using multiple features:")
print("Input:", sample_input_multi)
print("Predicted Output:", prediction_multi)