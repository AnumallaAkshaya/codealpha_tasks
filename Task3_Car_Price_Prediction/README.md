
TASK 3: Car Price Prediction with Machine Learning
This project implements Car Price Prediction using Machine Learning. The dataset contains information about used cars, including car age, present price, kilometers driven, fuel type, selling type, transmission, and previous owners. A Random Forest Regressor is trained to predict the selling price of a car based on these features.

The project includes data loading, data exploration, data preprocessing, feature engineering, categorical feature encoding, model training, prediction, visualization, and performance evaluation.

🛠️ Libraries Used
Library	Purpose
Pandas	Used for loading, cleaning, and analyzing the dataset
NumPy	Used for numerical calculations
Matplotlib	Used for creating and displaying charts
Scikit-learn	Used for preprocessing, splitting data, building the ML model, and evaluating performance
Scikit-learn Modules Used

train_test_split – Splits the dataset into training and testing data.

ColumnTransformer – Applies different preprocessing methods to numerical and categorical features.

OneHotEncoder – Converts categorical features into numerical values.

Pipeline – Combines preprocessing and model training into a single workflow.

RandomForestRegressor – Builds the Random Forest regression model for predicting car prices.

mean_absolute_error – Calculates the average absolute difference between actual and predicted prices.

mean_squared_error – Calculates the average squared difference between actual and predicted prices.

r2_score – Measures how well the model explains the variation in car prices.

📂 Dataset

The project uses the car dataset stored as:

car data.csv

The dataset contains information about used cars with the following features:

Car_Name – Name of the car
Year – Manufacturing year
Selling_Price – Selling price of the car
Present_Price – Current market price
Driven_kms – Number of kilometers driven
Fuel_Type – Fuel used by the car
Selling_type – Dealer or individual seller
Transmission – Manual or automatic
Owner – Number of previous owners

The target variable is:

Selling_Price

The model predicts the selling price of a car using its other characteristics.

🔍 Code Explanation
1. Importing Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

Imports the required libraries for data handling, numerical calculations, and visualization.

2. Importing Machine Learning Functions
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

Imports the Scikit-learn functions required for data splitting, preprocessing, model creation, and model evaluation.

3. Loading the Dataset
df = pd.read_csv("car data.csv")

Loads the car data.csv dataset into a Pandas DataFrame.

4. Exploring the Dataset
print("Dataset Shape:", df.shape)
print(df.head())
print(df.info())
print(df.isnull().sum())

These commands are used to:

View the number of rows and columns.
Display the first five records.
Check the structure and data types.
Check for missing values.
5. Removing Duplicate Records
df = df.drop_duplicates()

Removes duplicate records from the dataset to improve data quality.

6. Feature Engineering
df["Car_Age"] = 2026 - df["Year"]

Creates a new feature called Car_Age.

It calculates the age of the car based on its manufacturing year.

For example:

Year = 2018
Car Age = 2026 - 2018 = 8 years

This feature helps the model understand how the age of a car affects its selling price.

7. Separating Features and Target
X = df.drop("Selling_Price", axis=1)
y = df["Selling_Price"]

X contains the input features used for prediction.

y contains the target variable that the model needs to predict.

The target variable is:

Selling_Price
8. Removing Unnecessary Column
X = X.drop("Car_Name", axis=1)

The Car_Name column is removed because the model uses the numerical and categorical characteristics of the car rather than the individual car name.

9. Defining Numerical and Categorical Features
categorical_features = ["Fuel_Type", "Selling_type", "Transmission"]


numerical_features = [
    "Year",
    "Present_Price",
    "Driven_kms",
    "Owner",
    "Car_Age"
]

The numerical and categorical columns are separated so that they can be processed appropriately.

10. Encoding Categorical Features
preprocessor = ColumnTransformer(
    transformers=[
        ("num", "passthrough", numerical_features),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features)
    ]
)

Categorical values such as:

Petrol
Diesel
CNG

and:

Manual
Automatic

are converted into numerical values using OneHotEncoder.

11. Creating the Random Forest Regression Model
model = RandomForestRegressor(
    n_estimators=200,
    random_state=42
)

Creates a Random Forest Regressor using 200 decision trees.

The regression model is used because the target variable, Selling_Price, is a continuous numerical value.

12. Creating the Machine Learning Pipeline
pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("model", model)
    ]
)

Combines the preprocessing steps and Random Forest model into a single machine learning pipeline.

13. Splitting Training and Testing Data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

The dataset is divided into:

80% training data
20% testing data

The model learns from the training data and is evaluated using the testing data.

14. Training the Model
pipeline.fit(X_train, y_train)

Trains the Random Forest regression model using the training dataset.

The preprocessing and encoding are automatically performed before training.

15. Making Predictions
y_pred = pipeline.predict(X_test)

Uses the trained model to predict car prices for the test dataset.

16. Evaluating the Model
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

The model is evaluated using:

Mean Absolute Error

Measures the average difference between actual and predicted prices.

Mean Squared Error

Measures the average squared difference between actual and predicted prices.

Root Mean Squared Error

Calculates the square root of the Mean Squared Error.

R² Score

Measures how well the model explains the variation in car prices.

17. Displaying Model Results
print("Mean Absolute Error:", round(mae, 2))
print("Mean Squared Error:", round(mse, 2))
print("Root Mean Squared Error:", round(rmse, 2))
print("R2 Score:", round(r2, 2))

Displays the performance metrics of the trained model in the terminal.

18. Comparing Actual and Predicted Prices
results = pd.DataFrame({
    "Actual Price": y_test.values,
    "Predicted Price": y_pred
})


print(results.head(10))

Creates a table containing actual selling prices and prices predicted by the machine learning model.

19. Actual vs Predicted Price Visualization
plt.scatter(y_test, y_pred)

Creates a scatter plot comparing the actual car prices with the predicted car prices.

The graph helps visualize how closely the predictions match the actual prices.

The graph is titled:

Actual vs Predicted Car Prices
20. Car Price Distribution
plt.hist(y, bins=20, edgecolor="black")

Creates a histogram showing the distribution of selling prices in the dataset.

The graph helps understand how car prices are distributed.

21. Present Price vs Selling Price
plt.scatter(df["Present_Price"], df["Selling_Price"])

Creates a scatter plot showing the relationship between the current market price and selling price of cars.

This helps identify how present price affects the selling price.

22. Driven Kilometers vs Selling Price
plt.scatter(df["Driven_kms"], df["Selling_Price"])

Creates a scatter plot showing the relationship between kilometers driven and selling price.

This helps analyze whether cars with higher mileage tend to have lower selling prices.

23. Testing with a Sample Car
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

A sample car's details are given to the trained model.

The model predicts the expected selling price based on the provided features.

📊 Output

The project generates the following visualizations:

Actual vs Predicted Car Prices
Distribution of Car Prices
Present Price vs Selling Price
Driven Kilometers vs Selling Price

The terminal displays:

Dataset shape
First few records
Dataset information
Missing values
Model evaluation metrics
Actual vs predicted prices
Sample car prediction
Example Evaluation Metrics
Mean Absolute Error: ...
Mean Squared Error: ...
Root Mean Squared Error: ...
R2 Score: ...

The exact values may vary slightly depending on the dataset and model configuration.

🎯 Project Objective

The main objective of this project is to understand how Machine Learning can be applied to real-world car price prediction.

The project demonstrates:

Data preprocessing
Feature engineering
Categorical data encoding
Regression
Random Forest
Model evaluation
Data visualization
Price prediction

This project provides practical experience in using Python, Pandas, NumPy, Matplotlib, and Scikit-learn for a real-world machine learning problem.
