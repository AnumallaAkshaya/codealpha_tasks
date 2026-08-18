import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

df = pd.read_csv("car data.csv")

print("Dataset Shape:", df.shape)
print("\nFirst 5 Rows:")
print(df.head())

print("\nDataset Information:")
print(df.info())

print("\nMissing Values:")
print(df.isnull().sum())

df = df.drop_duplicates()

df["Car_Age"] = 2026 - df["Year"]

X = df.drop("Selling_Price", axis=1)
y = df["Selling_Price"]

X = X.drop("Car_Name", axis=1)

categorical_features = ["Fuel_Type", "Selling_type", "Transmission"]
numerical_features = ["Year", "Present_Price", "Driven_kms", "Owner", "Car_Age"]

preprocessor = ColumnTransformer(
    transformers=[
        ("num", "passthrough", numerical_features),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features)
    ]
)

model = RandomForestRegressor(
    n_estimators=200,
    random_state=42
)

pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("model", model)
    ]
)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

pipeline.fit(X_train, y_train)

y_pred = pipeline.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("\nModel Evaluation:")
print("Mean Absolute Error:", round(mae, 2))
print("Mean Squared Error:", round(mse, 2))
print("Root Mean Squared Error:", round(rmse, 2))
print("R2 Score:", round(r2, 2))

results = pd.DataFrame({
    "Actual Price": y_test.values,
    "Predicted Price": y_pred
})

print("\nActual vs Predicted Prices:")
print(results.head(10))

plt.figure(figsize=(8, 5))
plt.scatter(y_test, y_pred)
plt.xlabel("Actual Selling Price")
plt.ylabel("Predicted Selling Price")
plt.title("Actual vs Predicted Car Prices")
plt.grid(True)
plt.show()

plt.figure(figsize=(10, 5))
plt.hist(y, bins=20, edgecolor="black")
plt.xlabel("Selling Price")
plt.ylabel("Number of Cars")
plt.title("Distribution of Car Prices")
plt.grid(True)
plt.show()

plt.figure(figsize=(10, 5))
plt.scatter(df["Present_Price"], df["Selling_Price"])
plt.xlabel("Present Price")
plt.ylabel("Selling Price")
plt.title("Present Price vs Selling Price")
plt.grid(True)
plt.show()

plt.figure(figsize=(10, 5))
plt.scatter(df["Driven_kms"], df["Selling_Price"])
plt.xlabel("Driven Kilometers")
plt.ylabel("Selling Price")
plt.title("Driven Kilometers vs Selling Price")
plt.grid(True)
plt.show()

sample_car = pd.DataFrame({
    "Year": [2018],
    "Present_Price": [8.5],
    "Driven_kms": [25000],
    "Fuel_Type": ["Petrol"],
    "Selling_type": ["Dealer"],
    "Transmission": ["Manual"],
    "Owner": [0],
    "Car_Age": [2026 - 2018]
})

predicted_price = pipeline.predict(sample_car)

print("\nSample Car Details:")
print(sample_car)

print("\nPredicted Selling Price:", round(predicted_price[0], 2), "lakhs")