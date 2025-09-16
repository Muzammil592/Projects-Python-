import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

# Load dataset
url = "C:/Muzammil Abbas/AI models/Model 1 (Linear Regression)/dataset.csv"
df = pd.read_csv(url)

# Features (X) and target (y)
X = df.drop('medv', axis=1)
y = df['medv']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestRegressor(n_estimators=200, random_state=42)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print(f"Model trained ✅  |  RMSE: {rmse:.2f}  |  R²: {r2:.2f}")

# ==========================
# Take user input & predict
# ==========================

# Get input from user
print("\nEnter details of the house:")
crim = float(input("Crime rate: "))
zn = float(input("Residential land zone %: "))
indus = float(input("Industrial area %: "))
chas = int(input("Near Charles river? (1=yes, 0=no): "))
nox = float(input("Pollution (nox): "))
rm = float(input("Average number of rooms: "))
age = float(input("Old houses %: "))
dis = float(input("Distance to employment centers: "))
rad = int(input("Highway access index: "))
tax = float(input("Property tax rate: "))
ptratio = float(input("Student-teacher ratio: "))
black = float(input("Black population index: "))
lstat = float(input("Lower income %: "))

# Create data frame from input
user_data = pd.DataFrame([[
    crim, zn, indus, chas, nox, rm, age, dis, rad, tax, ptratio, black, lstat
]], columns=X.columns)

# Predict price
predicted_price = model.predict(user_data)[0]
print(f"\n🏡 Estimated House Price: ${predicted_price*1000:.2f}")
