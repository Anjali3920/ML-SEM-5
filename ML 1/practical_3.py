import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score


# ==========================================================
# 1. READ FAOSTAT CSV FILE
# ==========================================================

df = pd.read_csv("maize_japan.csv")

print("Original Data")
print("----------------")
print(df.head())

print("\nColumns")
print("----------------")
print(df.columns)


# ==========================================================
# 2. SELECT REQUIRED ELEMENTS
# ==========================================================

# We use:
# Area harvested
# Yield

df = df[
    df["Element"].isin([
        "Area harvested",
        "Yield"
    ])
]


# ==========================================================
# 3. CONVERT LONG FORMAT TO WIDE FORMAT
# ==========================================================

df_wide = df.pivot(
    index="Year",
    columns="Element",
    values="Value"
).reset_index()


# ==========================================================
# 4. RENAME COLUMNS
# ==========================================================

df_wide = df_wide.rename(columns={
    "Area harvested": "Area_Harvested",
    "Yield": "Yield"
})


# ==========================================================
# 5. SORT DATA BY YEAR
# ==========================================================

df_wide = df_wide.sort_values("Year")


# ==========================================================
# 6. CHECK PROCESSED DATA
# ==========================================================

print("\nProcessed Data")
print("----------------")
print(df_wide.head())

print("\nMissing Values")
print("----------------")
print(df_wide.isnull().sum())


# ==========================================================
# 7. REMOVE MISSING VALUES
# ==========================================================

df_wide = df_wide.dropna()


# ==========================================================
# 8. DEFINE INPUT AND OUTPUT
# ==========================================================

# Independent variables
X = df_wide[
    ["Year", "Area_Harvested"]
]

# Dependent variable
y = df_wide["Yield"]


# ==========================================================
# 9. CREATE MULTIVARIATE LINEAR REGRESSION MODEL
# ==========================================================

model = LinearRegression()


# ==========================================================
# 10. TRAIN MODEL
# ==========================================================

model.fit(X, y)


# ==========================================================
# 11. GET COEFFICIENTS
# ==========================================================

intercept = model.intercept_

year_coefficient = model.coef_[0]

area_coefficient = model.coef_[1]


# ==========================================================
# 12. PRINT REGRESSION EQUATION
# ==========================================================

print("\nMultivariate Linear Regression Equation")
print("----------------------------------------")

print(f"Intercept = {intercept:.4f}")

print(
    f"Year coefficient = {year_coefficient:.4f}"
)

print(
    f"Area Harvested coefficient = {area_coefficient:.4f}"
)

print("\nEquation:")

print(
    f"Yield = {intercept:.4f} "
    f"+ ({year_coefficient:.4f} × Year) "
    f"+ ({area_coefficient:.4f} × Area_Harvested)"
)


# ==========================================================
# 13. PREDICT HISTORICAL YIELD
# ==========================================================

y_pred = model.predict(X)


# ==========================================================
# 14. MODEL EVALUATION
# ==========================================================

mse = mean_squared_error(y, y_pred)

rmse = np.sqrt(mse)

r2 = r2_score(y, y_pred)


print("\nModel Evaluation")
print("----------------")

print(f"MSE  = {mse:.4f}")

print(f"RMSE = {rmse:.4f}")

print(f"R²   = {r2:.4f}")


# ==========================================================
# 15. HISTORICAL PREDICTION TABLE
# ==========================================================

results = pd.DataFrame({
    "Year": df_wide["Year"],
    "Actual_Yield": df_wide["Yield"],
    "Predicted_Yield": y_pred
})


print("\nHistorical Predictions")
print("----------------------")

print(results.head(10))


# ==========================================================
# 16. FUTURE YEARS
# ==========================================================

future_years = pd.DataFrame({
    "Year": [2027, 2028, 2029, 2030]
})


# ==========================================================
# 17. ESTIMATE FUTURE AREA HARVESTED
# ==========================================================

# Future Area Harvested is not available yet.
# Therefore, we estimate it using a separate
# linear regression model.

area_model = LinearRegression()

area_model.fit(
    df_wide[["Year"]],
    df_wide["Area_Harvested"]
)


# Predict Area Harvested for 2027-2030

future_area = area_model.predict(
    future_years[["Year"]]
)


future_years["Area_Harvested"] = future_area


print("\nEstimated Future Area Harvested")
print("--------------------------------")

print(
    future_years.to_string(index=False)
)


# ==========================================================
# 18. PREDICT FUTURE YIELD
# ==========================================================

future_predictions = model.predict(
    future_years[
        ["Year", "Area_Harvested"]
    ]
)


# ==========================================================
# 19. CREATE FUTURE PREDICTION TABLE
# ==========================================================

future = pd.DataFrame({
    "Year": future_years["Year"],
    "Estimated_Area_Harvested":
        future_years["Area_Harvested"],
    "Predicted_Yield":
        future_predictions
})


print("\nFuture Yield Prediction")
print("-----------------------")

print(
    future.to_string(index=False)
)


# ==========================================================
# 20. SAVE FUTURE PREDICTIONS
# ==========================================================

future.to_csv(
    "multivariate_yield_prediction_2027_2030.csv",
    index=False
)

print("\nPrediction file saved as:")
print(
    "multivariate_yield_prediction_2027_2030.csv"
)


# ==========================================================
# 21. PLOT HISTORICAL + FUTURE PREDICTIONS
# ==========================================================

plt.figure(figsize=(12, 7))


# ----------------------------------------------------------
# Historical actual yield
# ----------------------------------------------------------

plt.scatter(
    df_wide["Year"],
    df_wide["Yield"],
    color="blue",
    s=50,
    label="Historical Data",
    zorder=3
)


# ----------------------------------------------------------
# Historical model prediction
# ----------------------------------------------------------

plt.plot(
    df_wide["Year"],
    y_pred,
    color="green",
    linewidth=2.5,
    label="Multivariate Regression",
    zorder=2
)


# ----------------------------------------------------------
# Future prediction
# ----------------------------------------------------------

plt.plot(
    future["Year"],
    future["Predicted_Yield"],
    color="red",
    marker="o",
    markersize=10,
    linewidth=3,
    linestyle="--",
    label="Future Prediction (2027-2030)",
    zorder=5
)


# ==========================================================
# 22. LABEL FUTURE PREDICTIONS
# ==========================================================

for i in range(len(future)):

    year = future["Year"].iloc[i]

    prediction = future["Predicted_Yield"].iloc[i]

    plt.annotate(
        f"{prediction:.1f}",
        xy=(year, prediction),
        xytext=(0, 12),
        textcoords="offset points",
        ha="center",
        color="red",
        fontsize=10,
        fontweight="bold"
    )


# ==========================================================
# 23. SHOW WHERE FUTURE PREDICTION STARTS
# ==========================================================

plt.axvline(
    x=2025,
    color="gray",
    linestyle=":",
    linewidth=2,
    label="Future period"
)


# ==========================================================
# 24. GRAPH LABELS
# ==========================================================

plt.xlabel(
    "Year",
    fontsize=13
)

plt.ylabel(
    "Yield (kg/ha)",
    fontsize=13
)

plt.title(
    "Maize Yield Prediction Using "
    "Multivariate Linear Regression",
    fontsize=15
)


# ==========================================================
# 25. SHOW R² ON GRAPH
# ==========================================================

plt.text(
    0.03,
    0.95,
    f"R² = {r2:.4f}\nRMSE = {rmse:.2f}",
    transform=plt.gca().transAxes,
    verticalalignment="top",
    fontsize=11,
    bbox=dict(
        boxstyle="round",
        facecolor="white",
        alpha=0.8
    )
)


# ==========================================================
# 26. FINISH GRAPH
# ==========================================================

plt.legend()

plt.grid(
    True,
    alpha=0.3
)

plt.tight_layout()

plt.show()




