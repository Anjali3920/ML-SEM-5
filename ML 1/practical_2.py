import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# ==========================================
# 1. Read data
# ==========================================

df = pd.read_csv("crop_yield.csv")

print(df.head())
print(df.columns)


# ==========================================
# 2. Clean data
# ==========================================

df["Year"] = pd.to_numeric(df["Year"], errors="coerce")
df["Value"] = pd.to_numeric(df["Value"], errors="coerce")

df = df.dropna()
df = df.sort_values("Year")


# ==========================================
# 3. X and y
# ==========================================

X = df[["Year"]]
y = df["Value"]


# ==========================================
# 4. Polynomial transformation
# ==========================================

degree = 3

poly = PolynomialFeatures(degree=degree)

X_poly = poly.fit_transform(X)


# ==========================================
# 5. Train model
# ==========================================

model = LinearRegression()

model.fit(X_poly, y)


# ==========================================
# 6. Historical predictions
# ==========================================

y_pred = model.predict(X_poly)


# ==========================================
# 7. Model evaluation
# ==========================================

mse = mean_squared_error(y, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y, y_pred)

print("\nPolynomial Regression")
print("=====================")
print("Degree:", degree)

print("\nModel Evaluation")
print("----------------")
print("MSE :", mse)
print("RMSE:", rmse)
print("R²  :", r2)


# ==========================================
# 8. Future years
# ==========================================

future_years = pd.DataFrame({
    "Year": [2027, 2028, 2029, 2030]
})


# ==========================================
# 9. Polynomial future features
# ==========================================

future_poly = poly.transform(future_years)


# ==========================================
# 10. Future predictions
# ==========================================

future_predictions = model.predict(future_poly)


future = pd.DataFrame({
    "Year": [2027, 2028, 2029, 2030],
    "Predicted_Yield": future_predictions
})


print("\nPredicted Yield")
print("----------------")
print(future.to_string(index=False))


# ==========================================
# 11. Save predictions
# ==========================================

future.to_csv(
    "polynomial_prediction_2027_2030.csv",
    index=False
)


# ==========================================
# 12. Create smooth curve
# ==========================================

line_years = np.arange(
    int(df["Year"].min()),
    2031
)

line_df = pd.DataFrame({
    "Year": line_years
})

line_poly = poly.transform(line_df)

line_predictions = model.predict(line_poly)


# ==========================================
# 13. Plot
# ==========================================

plt.figure(figsize=(12, 7))


# Historical data
plt.scatter(
    df["Year"],
    df["Value"],
    color="blue",
    s=45,
    label="Historical Data"
)


# Polynomial curve
plt.plot(
    line_years,
    line_predictions,
    color="green",
    linewidth=2.5,
    label="Polynomial Regression (Degree 2)"
)


# Future predictions
plt.scatter(
    future["Year"],
    future["Predicted_Yield"],
    color="red",
    s=90,
    zorder=5,
    label="Predicted Yield (2027–2030)"
)


# Future prediction line
plt.plot(
    future["Year"],
    future["Predicted_Yield"],
    color="red",
    linestyle="--",
    linewidth=2
)


# Prediction labels
for _, row in future.iterrows():

    plt.annotate(
        f'{row["Predicted_Yield"]:.1f}',
        (row["Year"], row["Predicted_Yield"]),
        xytext=(0, 10),
        textcoords="offset points",
        ha="center"
    )


plt.xlabel("Year", fontsize=12)
plt.ylabel("Yield", fontsize=12)

plt.title(
    "Crop Yield Prediction Using Polynomial Regression",
    fontsize=15
)

plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()

plt.show()

