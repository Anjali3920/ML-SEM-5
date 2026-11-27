# ==========================================
# KNN - MUSHROOM EDIBILITY CLASSIFICATION
# ==========================================

# 1. Import libraries
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


# ==========================================
# 2. Load Dataset
# ==========================================

df = pd.read_csv("11_mushroom_edibility.csv")

print("First 5 rows:")
print(df.head())

print("\nDataset shape:")
print(df.shape)


# ==========================================
# 3. Remove SampleID
# ==========================================

# SampleID is only an identifier.
# It should not be used for prediction.

df = df.drop("SampleID", axis=1)


# ==========================================
# 4. Separate Features and Target
# ==========================================

X = df.drop("Class", axis=1)
y = df["Class"]


print("\nFeatures:")
print(X.columns.tolist())

print("\nTarget:")
print(y.name)


# ==========================================
# 5. Check Class Distribution
# ==========================================

print("\nClass Distribution:")
print(y.value_counts())


# ==========================================
# 6. Encode Target
# ==========================================

# poisonous / edible → numerical values

label_encoder = LabelEncoder()

y = label_encoder.fit_transform(y)

print("\nEncoded Classes:")
print(label_encoder.classes_)


# ==========================================
# 7. Train-Test Split
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


print("\nTraining samples:", len(X_train))
print("Testing samples:", len(X_test))


# ==========================================
# 8. One-Hot Encoding
# ==========================================

# All our input features are categorical.

categorical_columns = X.columns

preprocessor = ColumnTransformer(
    transformers=[
        (
            "categorical",
            OneHotEncoder(handle_unknown="ignore"),
            categorical_columns
        )
    ]
)


# Fit encoder only on training data
X_train = preprocessor.fit_transform(X_train)

# Transform test data
X_test = preprocessor.transform(X_test)


# ==========================================
# 9. Create KNN Model
# ==========================================

k = 5

knn = KNeighborsClassifier(n_neighbors=k)


# ==========================================
# 10. Train Model
# ==========================================

knn.fit(X_train, y_train)


# ==========================================
# 11. Make Predictions
# ==========================================

y_pred = knn.predict(X_test)


# ==========================================
# 12. Accuracy
# ==========================================

accuracy = accuracy_score(y_test, y_pred)

print("\n==============================")
print("KNN RESULTS")
print("==============================")

print("\nK value:", k)

print("Accuracy:", accuracy)

print("Accuracy (%):", accuracy * 100)


# ==========================================
# 13. Confusion Matrix
# ==========================================

cm = confusion_matrix(y_test, y_pred)

print("\nConfusion Matrix:")
print(cm)


# ==========================================
# 14. Classification Report
# ==========================================

print("\nClassification Report:")
print(classification_report(
    y_test,
    y_pred,
    target_names=label_encoder.classes_
))