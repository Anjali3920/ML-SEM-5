# ==========================================
# SVM - MUSHROOM EDIBILITY CLASSIFICATION
# ==========================================

# 1. Import libraries

import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from sklearn.svm import SVC

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report
)


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
# 5. Encode Target
# ==========================================

label_encoder = LabelEncoder()

y = label_encoder.fit_transform(y)

print("\nClasses:")
print(label_encoder.classes_)


# ==========================================
# 6. Train-Test Split
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
# 7. Identify Categorical Columns
# ==========================================

categorical_columns = X.columns


# ==========================================
# 8. One-Hot Encoding
# ==========================================

preprocessor = ColumnTransformer(
    transformers=[
        (
            "categorical",
            OneHotEncoder(handle_unknown="ignore"),
            categorical_columns
        )
    ]
)


# ==========================================
# 9. Create SVM Pipeline
# ==========================================

model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),

        ("scaler", StandardScaler(with_mean=False)),

        ("svm", SVC(
            kernel="rbf",
            C=1.0
        ))
    ]
)


# ==========================================
# 10. Train SVM
# ==========================================

model.fit(X_train, y_train)


# ==========================================
# 11. Prediction
# ==========================================

y_pred = model.predict(X_test)


# ==========================================
# 12. Accuracy
# ==========================================

accuracy = accuracy_score(y_test, y_pred)

print("\n==============================")
print("SVM RESULTS")
print("==============================")

print("\nAccuracy:", accuracy)

print("Accuracy (%):", accuracy * 100)


# ==========================================
# 13. Confusion Matrix
# ==========================================

print("\nConfusion Matrix:")

cm = confusion_matrix(y_test, y_pred)

print(cm)


# ==========================================
# 14. Classification Report
# ==========================================

print("\nClassification Report:")

print(
    classification_report(
        y_test,
        y_pred,
        target_names=label_encoder.classes_
    )
)