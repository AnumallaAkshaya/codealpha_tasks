

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)



df = pd.read_csv("Iris.csv")

print("\nFirst 5 Rows")
print(df.head())

print("\nDataset Info")
print(df.info())

print("\nMissing Values")
print(df.isnull().sum())


df.drop("Id", axis=1, inplace=True)



print("\nStatistical Summary")
print(df.describe())

# Species distribution chart
plt.figure(figsize=(6,4))
sns.countplot(x="Species", data=df)
plt.title("Species Distribution")
plt.savefig("species_distribution.png")
plt.show()

# Pairplot
sns.pairplot(df, hue="Species")
plt.savefig("pairplot.png")
plt.show()



X = df.drop("Species", axis=1)
y = df["Species"]



X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)



model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

model.fit(X_train, y_train)


y_pred = model.predict(X_test)


accuracy = accuracy_score(y_test, y_pred)

print("\nAccuracy:")
print(f"{accuracy*100:.2f}%")

print("\nClassification Report:")
print(classification_report(y_test, y_pred))



cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(6,5))
sns.heatmap(
    cm,
    annot=True,
    cmap="Blues",
    fmt="d"
)

plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")

plt.savefig("confusion_matrix.png")
plt.show()



sample_flower = [[5.1, 3.5, 1.4, 0.2]]

prediction = model.predict(sample_flower)

print("\nSample Prediction:")
print("Flower Species:", prediction[0])

joblib.dump(model, "iris_model.pkl")

print("\nModel saved as iris_model.pkl")
print("Task Completed Successfully!")
