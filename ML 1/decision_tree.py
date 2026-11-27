# ==========================================
# DECISION TREE - LOAN APPROVAL DATASET
# ==========================================

# Step 1: Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report
)

# Step 2: Load dataset
df = pd.read_csv("07_loan_approval.csv")

# Step 3: Display dataset
print("First 5 rows:")
print(df.head())

print("\nDataset Shape:")
print(df.shape)

print("\nDataset Information:")
print(df.info())

# Step 4: Check missing values
print("\nMissing Values:")
print(df.isnull().sum())

# Step 5: Target distribution
print("\nLoan Approval Distribution:")
print(df["LoanApproved"].value_counts())

# Step 6: Separate features and target
X = df.drop(["ApplicantID", "LoanApproved"], axis=1)
y = df["LoanApproved"]

# Step 7: Convert categorical variables into numerical variables
X = pd.get_dummies(X, drop_first=True)

# Step 8: Encode target variable
le = LabelEncoder()
y = le.fit_transform(y)

print("\nTarget Classes:")
print(le.classes_)

# Step 9: Split data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining Data:", X_train.shape)
print("Testing Data:", X_test.shape)

# Step 10: Create Decision Tree
model = DecisionTreeClassifier(
    criterion="gini",
    random_state=42
)

# Step 11: Train model
model.fit(X_train, y_train)

# Step 12: Prediction
y_pred = model.predict(X_test)

# Step 13: Accuracy
accuracy = accuracy_score(y_test, y_pred)

print("\nAccuracy:")
print(accuracy)

print("\nAccuracy Percentage:")
print(accuracy * 100)

# Step 14: Confusion Matrix
cm = confusion_matrix(y_test, y_pred)

print("\nConfusion Matrix:")
print(cm)

# Step 15: Classification Report
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Step 16: Plot Decision Tree
plt.figure(figsize=(20, 10))

plot_tree(
    model,
    feature_names=X.columns,
    class_names=le.classes_,
    filled=True,
    rounded=True
)

plt.title("Decision Tree for Loan Approval")
plt.show()

# Step 17: Feature Importance
feature_importance = pd.DataFrame({
    "Feature": X.columns,
    "Importance": model.feature_importances_
})

feature_importance = feature_importance.sort_values(
    by="Importance",
    ascending=False
)

print("\nFeature Importance:")
print(feature_importance)

# Step 18: Feature Importance Graph
plt.figure(figsize=(10, 6))

plt.barh(
    feature_importance["Feature"],
    feature_importance["Importance"]
)

plt.xlabel("Importance")
plt.ylabel("Feature")
plt.title("Feature Importance")

plt.gca().invert_yaxis()

plt.show()