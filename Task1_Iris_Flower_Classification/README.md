
TASK 1: Iris Flower Classification 

This project implements Iris Flower Classification using Machine Learning. The Iris dataset contains measurements of iris flowers, including sepal length, sepal width, petal length, and petal width. A Random Forest Classifier is trained to classify flowers into three species: Setosa, Versicolor, and Virginica.

The project includes data loading, data exploration, visualization, model training, prediction, performance evaluation, and saving the trained model.

🛠️ Libraries Used
Library	Purpose
Pandas	Used for loading, cleaning, and analyzing the dataset
Seaborn	Used for creating statistical visualizations
Matplotlib	Used for creating and displaying charts
Scikit-learn	Used for splitting data, building the ML model, and evaluating performance
Joblib	Used for saving the trained machine learning model
Scikit-learn Modules Used
train_test_split – Splits the dataset into training and testing data.
RandomForestClassifier – Builds the Random Forest classification model.
accuracy_score – Calculates the model's accuracy.
classification_report – Provides precision, recall, F1-score, and support.
confusion_matrix – Shows the correctly and incorrectly classified samples.

Dataset:
The project uses the Iris dataset stored as:

Iris.csv

The dataset contains four flower measurements:

Sepal Length
Sepal Width
Petal Length
Petal Width

The target variable is:

Species

with three classes:

Iris-setosa
Iris-versicolor
Iris-virginica
🔍 Code Explanation
1. Importing Libraries
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import joblib

Imports the required libraries for data handling, visualization, machine learning model saving, and analysis.

2. Importing Machine Learning Functions
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

Imports the Scikit-learn functions required for data splitting, model creation, and model evaluation.

3. Loading the Dataset
df = pd.read_csv("Iris.csv")

Loads the Iris.csv dataset into a Pandas DataFrame.

4. Exploring the Dataset
print(df.head())
print(df.info())
print(df.isnull().sum())

These commands are used to:

View the first five rows.
Check the structure and data types.
Check for missing values.
5. Removing Unnecessary Column
df.drop("Id", axis=1, inplace=True)

The Id column is removed because it does not provide useful information for predicting the flower species.

6. Statistical Analysis
print(df.describe())

Provides statistical information such as:

Mean
Standard deviation
Minimum value
Maximum value
Quartiles

for the numerical features.

7. Visualizing Species Distribution
sns.countplot(x="Species", data=df)

Creates a bar chart showing the number of samples belonging to each Iris species.

The chart is saved as:

species_distribution.png
8. Creating Pairplot
sns.pairplot(df, hue="Species")

Creates pairwise graphs between the flower measurements and uses different colors for each species.

This helps visualize how the three species differ based on their measurements.

The graph is saved as:

pairplot.png
9. Separating Features and Target
X = df.drop("Species", axis=1)
y = df["Species"]
X contains the flower measurements used as input features.
y contains the flower species that the model needs to predict.
10. Splitting Training and Testing Data
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

11. Creating the Random Forest Model
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

Creates a Random Forest Classifier using 100 decision trees.

12. Training the Model
model.fit(X_train, y_train)

Trains the Random Forest model using the training data.

13. Making Predictions
y_pred = model.predict(X_test)

Uses the trained model to predict the species of flowers in the test dataset.

14. Evaluating Accuracy
accuracy = accuracy_score(y_test, y_pred)

Calculates the percentage of test samples that were classified correctly.

15. Classification Report
print(classification_report(y_test, y_pred))

Displays:

Precision
Recall
F1-score
Support

These metrics provide a more detailed evaluation of the classification model.

16. Confusion Matrix
cm = confusion_matrix(y_test, y_pred)

Creates a confusion matrix to show how many flowers were correctly and incorrectly classified for each species.

The visualization is saved as:

confusion_matrix.png
17. Testing with a Sample Flower
sample_flower = pd.DataFrame(
    [[5.1, 3.5, 1.4, 0.2]],
    columns=[
        "SepalLengthCm",
        "SepalWidthCm",
        "PetalLengthCm",
        "PetalWidthCm"
    ]
)


prediction = model.predict(sample_flower)


print("Flower Species:", prediction[0])

A new flower's measurements are given to the trained model, which predicts its species.

18. Saving the Trained Model
joblib.dump(model, "iris_model.pkl")

Saves the trained Random Forest model as:

iris_model.pkl

This allows the trained model to be reused later without training it again.

📊 Output

The project generates:

species_distribution.png
pairplot.png
confusion_matrix.png
iris_model.pkl

The terminal displays:

Dataset information
Missing values
Statistical summary
Model accuracy
Classification report
Sample flower prediction
