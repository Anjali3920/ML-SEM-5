# ==========================================
# Crop Yield Prediction using Linear Regression
# ==========================================

import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error


# ==========================================
# Step 1: Load Dataset
# ==========================================

df = pd.read_csv("crop_data.csv")


print("\nFirst Five Rows")
print(df.head())

print("\nColumns")
print(df.columns)


# ==========================================
# Step 2: Select Input and Output
# ==========================================

# Input feature
X = df[['Year']]

# Target value
y = df['Value']


# ==========================================
# Step 3: Split Dataset
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)


print("\nTraining Samples:", len(X_train))
print("Testing Samples :", len(X_test))


# ==========================================
# Step 4: Train Linear Regression Model
# ==========================================

model = LinearRegression()

model.fit(
    X_train,
    y_train
)


# ==========================================
# Step 5: Find m and c
# ==========================================

m = model.coef_[0]
c = model.intercept_


print("\nRegression Equation")
print(
    f"Yield = {m:.4f} × Year + ({c:.4f})"
)



# ==========================================
# Step 6: Test Prediction
# ==========================================

y_pred = model.predict(X_test)
 Step 7: Model Evaluation
# ==========================================

r2 = r2_score(
    y_test,
    y_pred
)

mse = mean_squared_error(
    y_test,
    y_pred
)


print("\nModel Performance")

print("R² Score :", round(r2,4))

print("MSE :", round(mse,4))



# ==========================================
# Step 8: Actual vs Predicted Table
# ==========================================

result = pd.DataFrame({

    "Year": X_test["Year"],

    "Actual Yield": y_test,

    "Predicted Yield": y_pred

})


result = result.sort_values("Year")


print("\nActual vs Predicted")

print(result)



# ==========================================
# PLOT 1:
# Actual vs Predicted Test Data
# ==========================================


plt.figure(figsize=(8,5))


plt.scatter(
    result["Year"],
    result["Actual Yield"],
    color="blue",
    s=80,
    label="Actual Yield"
)


plt.scatter(
    result["Year"],
    result["Predicted Yield"],
    color="red",
    marker="x",
    s=100,
    label="Predicted Yield"
)


plt.xlabel("Year")

plt.ylabel("Yield")

plt.title(
    "Actual vs Predicted Yield"
)


plt.legend()

plt.grid(True)

plt.show()



# ==========================================
# Step 9: Future Prediction 2028-2030
# ==========================================


future_years = pd.DataFrame({

    "Year":[
        2028,
        2029,
        2030
    ]

})


future_predictions = model.predict(
    future_years
)



future_result = pd.DataFrame({

    "Year":future_years["Year"],

    "Predicted Yield":future_predictions

})


print("\nFuture Predictions")

print(future_result)



# ==========================================
# PLOT 2:
# Historical + Future Prediction
# ==========================================


plt.figure(figsize=(10,6))


# Historical data

plt.scatter(
    df["Year"],
    df["Value"],
    color="blue",
    label="Historical Data"
)



# Regression line till 2030

future_line_years = pd.DataFrame({

    "Year":range(
        df["Year"].min(),
        2031
    )

})


line_prediction = model.predict(
    future_line_years
)



plt.plot(
    future_line_years["Year"],
    line_prediction,
    color="red",
    linewidth=2,
    label="Regression Line"
)



# Future predicted points

plt.scatter(

    future_result["Year"],

    future_result["Predicted Yield"],

    color="green",

    marker="*",

    s=250,

    label="Future Prediction"

)



plt.xlabel("Year")

plt.ylabel("Yield")


plt.title(
    "Crop Yield Forecast (2028-2030)"
)


plt.legend()

plt.grid(True)


plt.show()
