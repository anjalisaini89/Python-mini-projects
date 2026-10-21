from google.colab import files 
uploaded = files.upload()
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import seaborn as sns
# 3. Read CSV
df = pd.read_csv("modified_data.csv")
print("Dataset Shape:", df.shape)
display(df.head())
# 4. Create classification target
median_price = df["price"].median()
df["price_class"] = (
df["price"] >= median_price
).astype(int)
print("Median Price:", median_price)
print("\nClass Distribution:")
print(df["price_class"].value_counts())
# 5. Separate X and y
X = df.drop(
columns=[
"price",
"price_class",
"price_per_sqft",
"date",
"street"
]
)
y = df["price_class"]
# 6. Identify categorical columns
categorical_columns = X.select_dtypes(
include="object"
).columns.tolist()
print("\nCategorical Columns:")
print(categorical_columns)
# 7. One-Hot Encoding
preprocessor = ColumnTransformer(
transformers=[
(
"categorical",
OneHotEncoder(handle_unknown="ignore"), categorical_columns
)
],
remainder="passthrough"
)
# 8. Random Forest
rf = RandomForestClassifier(
n_estimators=200,
random_state=42,
n_jobs=-1
)
# 9. Pipeline
model = Pipeline(
steps=[
("preprocessor", preprocessor),
("classifier", rf)
]
)
# 10. Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
X,
y,
test_size=0.20,
random_state=42,
stratify=y
)
print("\nTraining Data:", X_train.shape)
print("Testing Data:", X_test.shape)
# 11. Train Model
model.fit(X_train, y_train)
# 12. Prediction
y_pred = model.predict(X_test)
# 13. Accuracy
accuracy = accuracy_score(y_test, y_pred)
print("\nAccuracy:", accuracy)
print("Accuracy (%):", accuracy * 100)
# 14. Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
print("\nConfusion Matrix:")
print(cm)
# 15. Plot Confusion Matrix
plt.figure(figsize=(6, 5))
sns.heatmap(
cm,
annot=True,
fmt="d",
cmap="Blues",
xticklabels=["Low Price", "High Price"],
yticklabels=["Low Price", "High Price"]
)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Random Forest Classification")
plt.show()
# 16. Classification Report
print("\nClassification Report:")
print(
classification_report(
y_test,
y_pred,
target_names=["Low Price", "High Price"]
)
)

