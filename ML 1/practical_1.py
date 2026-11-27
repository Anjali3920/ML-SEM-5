


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score


# ==========================================
# 1. Read the data
# ==========================================

df = pd.read_csv("crop_yield.csv")

# Make sure Year and Value are numeric
df["Year"] = pd.to_numeric(df["Year"], errors="coerce")
df["Value"] = pd.to_numeric(df["Value"], errors="coerce")

# Remove missing values
df = df.dropna()

# Sort by year
df = df.sort_values("Year")

print("Data:")
print(df)


# ==========================================
# 2. Prepare data for Linear Regression
# ==========================================

X = df[["Year"]]
y = df["Value"]


# ==========================================
# 3. Train Linear Regression model
# ==========================================

model = LinearRegression()
model.fit(X, y)


# ==========================================
# 4. Regression equation
# ==========================================

m = model.coef_[0]
c = model.intercept_

print("\nLinear Regression Equation")
print("--------------------------")
print(f"Slope (m)     = {m:.4f}")
print(f"Intercept (c) = {c:.4f}")
print(f"Equation      = Yield = {m:.4f} × Year + ({c:.4f})")


# ==========================================
# 5. Historical predictions
# ==========================================

y_pred = model.predict(X)


# ==========================================
# 6. Model evaluation
# ==========================================

mse = mean_squared_error(y, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y, y_pred)

print("\nModel Evaluation")
print("--------------------------")
print(f"MSE  = {mse:.4f}")
print(f"RMSE = {rmse:.4f}")
print(f"R²   = {r2:.4f}")


# ==========================================
# 7. Predict 2027–2030
# ==========================================

future_years = pd.DataFrame({
    "Year": [2027, 2028, 2029, 2030]
})

future_predictions = model.predict(future_years)

future = pd.DataFrame({
    "Year": [2027, 2028, 2029, 2030],
    "Predicted_Yield": future_predictions
})


# ==========================================
# 8. Display predictions
# ==========================================

print("\nPredicted Yield")
print("--------------------------")
print(future.to_string(index=False))


# ==========================================
# 9. Save predictions
# ==========================================

future.to_csv(
    "yield_prediction_2027_2030.csv",
    index=False
)

print("\nPrediction file saved as:")
print("yield_prediction_2027_2030.csv")


# ==========================================
# 10. Create regression line
# ==========================================

# Years from first historical year to 2030
line_years = np.arange(
    int(df["Year"].min()),
    2031
)

line_df = pd.DataFrame({
    "Year": line_years
})

line_predictions = model.predict(line_df)


# ==========================================
# 11. Plot
# ==========================================

plt.figure(figsize=(12, 7))

# Historical data
plt.scatter(
    df["Year"],
    df["Value"],
    color="blue",
    s=45,
    label="Historical FAOSTAT Data"
)

# Linear regression line
plt.plot(
    line_years,
    line_predictions,
    color="green",
    linewidth=2,
    label="Linear Regression"
)

# 2027–2030 predictions
plt.scatter(
    future["Year"],
    future["Predicted_Yield"],
    color="red",
    s=90,
    zorder=5,
    label="Predicted Yield (2027–2030)"
)

# Connect prediction points
plt.plot(
    future["Year"],
    future["Predicted_Yield"],
    color="red",
    linestyle="--",
    linewidth=2
)

# Add values to prediction points
for _, row in future.iterrows():
    plt.annotate(
        f'{row["Predicted_Yield"]:.1f}',
        (row["Year"], row["Predicted_Yield"]),
        xytext=(0, 10),
        textcoords="offset points",
        ha="center",
        fontsize=10
    )


# Labels and title
plt.xlabel("Year", fontsize=12)
plt.ylabel("Yield", fontsize=12)

plt.title(
    "Crop Yield Prediction Using Linear Regression",
    fontsize=15
)

plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()

plt.show()

