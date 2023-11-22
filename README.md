# PolinomialRegression

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
import numpy as np

data = pd.read_csv("Position_Salaries.csv")

X = pd.DataFrame(data.Level).squeeze()
Y = pd.DataFrame(data.Salary).squeeze()

x = np.array(X).reshape(-1, 1)
y = np.array(Y).reshape(-1, 1)


poly_features = PolynomialFeatures(degree=3)
x_poly = poly_features.fit_transform(x)
poly_model = LinearRegression()
poly_model.fit(x_poly, y)

print(poly_model.score(x_poly, y))

new_temp = 13
new_temp_poly = poly_features.transform(np.array([[new_temp]]))
predicted_sales = poly_model.predict(new_temp_poly)
print(predicted_sales)
