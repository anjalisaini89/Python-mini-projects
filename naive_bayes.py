# NAIVE BAYES CLASSIFICATION
# Dataset: modified_data(1).csv

# Step 1: Import Libraries
import pandas as pd
import numpy as np

from google.colab import files
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report
)

# Step 2: Upload CSV File
uploaded = files.upload()
file_name = list(uploaded.keys())[0]
df = pd.read_csv(file_name)

print("Dataset loaded successfully!")
print("Dataset Shape:", df.shape)

# Step 3: Display Dataset
print("\nFirst 5 Rows:")
display(df.head())

print("\nColumn Names:")
print(df.columns.tolist())

# Step 4: Check Missing Values
print("\nMissing Values:")
print(df.isnull().sum())

# Step 5: Create Classification Target
median_price = df["price"].median()

df["price_category"] = np.where(
    df["price"] >= median_price, 1, 0
)

print("\nMedian Price:", median_price)
print("0 = Low Price")
print("1 = High Price")
print(df[["price", "price_category"]].head())

# Step 6: Select Features
features = [
    "bedrooms",
    "bathrooms",
    "sqft_living",
    "sqft_lot",
    "floors",
    "waterfront",
    "view",
    "condition",
    "sqft_above",
    "sqft_basement",
    "yr_built",
    "yr_renovated"
]

X = df[features]
y = df["price_category"]

# Step 7: Handle Missing Values
X = X.fillna(X.median())

# Step 8: Split Dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining Data Shape:", X_train.shape)
print("Testing Data Shape:", X_test.shape)

# Step 9: Create Naive Bayes Model
model = GaussianNB()

# Step 10: Train Model
model.fit(X_train, y_train)
print("\nNaive Bayes Model Trained Successfully!")

# Step 11: Make Predictions
y_pred = model.predict(X_test)

# Step 12: Display Predictions
print("\nActual Values:")
print(y_test.values[:20])

print("\nPredicted Values:")
print(y_pred[:20])

# Step 13: Calculate Accuracy
accuracy = accuracy_score(y_test, y_pred)

print("\n==============================")
print("MODEL ACCURACY")
print("==============================")
print("Accuracy:", accuracy)
print("Accuracy Percentage:", round(accuracy * 100, 2), "%")

# Step 14: Confusion Matrix
cm = confusion_matrix(y_test, y_pred)

print("\n==============================")
print("CONFUSION MATRIX")
print("==============================")
print(cm)

# Step 15: Classification Report
print("\n==============================")
print("CLASSIFICATION REPORT")
print("==============================")

print(classification_report(
    y_test,
    y_pred,
    target_names=["Low Price", "High Price"],
    zero_division=0
))

# Step 16: Predict a New House
new_house = pd.DataFrame([{
    "bedrooms": 3,
    "bathrooms": 2,
    "sqft_living": 1800,
    "sqft_lot": 5000,
    "floors": 1,
    "waterfront": 0,
    "view": 0,
    "condition": 3,
    "sqft_above": 1800,
    "sqft_basement": 0,
    "yr_built": 2005,
    "yr_renovated": 0
}])

prediction = model.predict(new_house[features])

print("\n==============================")
print("NEW HOUSE PREDICTION")
print("==============================")

if prediction[0] == 1:
    print("Prediction: HIGH PRICE")
else:
    print("Prediction: LOW PRICE")
