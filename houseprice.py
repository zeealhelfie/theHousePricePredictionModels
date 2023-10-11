# %%
import warnings
warnings.filterwarnings('ignore')
#https://www.kaggle.com/code/emrearslan123/house-price-prediction
# Import package
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from xgboost import XGBRegressor
from sklearn.preprocessing import PolynomialFeatures
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import BaggingRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score



# %%
data = pd.read_csv("house_data.csv")

# %%
print("Missing Values by Column")
print("-"*30)
print(data.isna().sum())
print("-"*30)
print("TOTAL MISSING VALUES:",data.isna().sum().sum())

# %%
import matplotlib.pyplot as plt

# Plot a histogram of the 'price' column
plt.hist(data['price'], bins=50)
plt.xlabel('Count')
plt.ylabel('Price')
plt.show()

# Plot a density plot of the 'price' column
plt.figure()
data['price'].plot(kind='density')
plt.xlabel('Price')
plt.show()


# %%
# Add a logarithmic transformation to the 'price' variable
data['price'] = np.log(data['price'])

# %%
# Plot a histogram of the 'price' column
plt.hist(data['price'], bins=50)
plt.xlabel('Count')
plt.ylabel('Price')
plt.show()

# Plot a density plot of the 'price' column
plt.figure()
data['price'].plot(kind='density')
plt.xlabel('Price')
plt.show()

# %%
print(data.info())

# %%
# scatter plor
pd.plotting.scatter_matrix(data, figsize=(10, 10))
plt.show()

# %%
#Pearson correlation coefficient
correlations = data.corr(method='pearson')
correlations

# %%
X = data[['sqft_living','bedrooms','bathrooms','sqft_lot','waterfront','view','condition','grade','sqft_above','zipcode','yr_built','yr_renovated','lat','long','sqft_living15','sqft_lot15']]
#X = data[['bathrooms', 'sqft_living', 'grade', 'sqft_above']]
y = data['price']

# Create a StandardScaler object
scaler = StandardScaler()
# Fit the scaler on the data and transform the data
X = scaler.fit_transform(X)


# %%
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# %%
def r2_cv(model):
    r2 = cross_val_score(model, X, y, scoring="r2", cv=5).mean()
    return r2

def evaluation(y, predictions):
    mae = mean_absolute_error(y, predictions)
    mse = mean_squared_error(y, predictions)
    #rmse = np.sqrt(mean_squared_error(y, predictions))
    r_squared = r2_score(y, predictions)
    return mae, mse, r_squared

# %%
models = pd.DataFrame(columns=["Model","MAE","MSE","R2 Score","R2 Score (Cross-Validation)"])

# %%
#Linear Regression
lin_reg = LinearRegression()
lin_reg.fit(X_train, y_train)
predictions = lin_reg.predict(X_test)

mae, mse, r_squared = evaluation(y_test, predictions)
print("MAE:", mae)
print("MSE:", mse)
print("R2 Score:", r_squared)
print("-"*30)
r2_cross_val = r2_cv(lin_reg)
print("R2 Score Cross-Validation:", r2_cross_val)

new_row = {"Model": "LinearRegression","MAE": mae, "MSE": mse, "R2 Score": r_squared, "R2 Score (Cross-Validation)": r2_cross_val}
models = models.append(new_row, ignore_index=True)

# %%
#Ridge Regression
ridge_alpha = 0.01
ridge = Ridge(alpha=ridge_alpha)
ridge.fit(X_train, y_train)
predictions = ridge.predict(X_test)

mae, mse, r_squared = evaluation(y_test, predictions)
print("MAE:", mae)
print("MSE:", mse)
print("R2 Score:", r_squared)
print("-"*30)
r2_cross_val = r2_cv(ridge)
print("R2 Score: Cross-Validation:", r2_cross_val)

new_row = {"Model": "Ridge","MAE": mae, "MSE": mse, "R2 Score": r_squared, "R2 Score (Cross-Validation)": r2_cross_val}
models = models.append(new_row, ignore_index=True)

# %%
#Lasso Regression 
alpha = 0.01
lasso = Lasso(alpha=0.01)
lasso.fit(X_train, y_train)
predictions = lasso.predict(X_test)

mae, mse, r_squared = evaluation(y_test, predictions)
print("MAE:", mae)
print("MSE:", mse)
print("R2 Score:", r_squared)
print("-"*30)
r2_cross_val = r2_cv(lasso)
print("R2 Score: Cross-Validation:", r2_cross_val)

new_row = {"Model": "Lasso","MAE": mae, "MSE": mse, "R2 Score": r_squared, "R2 Score (Cross-Validation)": r2_cross_val}
models = models.append(new_row, ignore_index=True)

# %%
#Elastic Net
elastic_net_alpha = 0.01
elastic_net = ElasticNet(alpha=elastic_net_alpha)
elastic_net.fit(X_train, y_train)
predictions = elastic_net.predict(X_test)

mae, mse, r_squared = evaluation(y_test, predictions)
print("MAE:", mae)
print("MSE:", mse)
print("R2 Score:", r_squared)
print("-"*30)
r2_cross_val = r2_cv(elastic_net)
print("R2 Score: Cross-Validation:", r2_cross_val)

new_row = {"Model": "ElasticNet","MAE": mae, "MSE": mse, "R2 Score": r_squared, "R2 Score (Cross-Validation)": r2_cross_val}
models = models.append(new_row, ignore_index=True)

# %%
#X = data[['bathrooms', 'sqft_living', 'grade', 'sqft_above']]
#y = data['price']
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# %%
#Random Forest Regressor
random_forest = RandomForestRegressor(n_estimators=100)
random_forest.fit(X_train, y_train)
predictions = random_forest.predict(X_test)

mae, mse, r_squared = evaluation(y_test, predictions)
print("MAE:", mae)
print("MSE:", mse)
print("R2 Score:", r_squared)
print("-"*30)
r2_cross_val = r2_cv(random_forest)
print("R2 Scores(Cross-Validation):", r2_cross_val)

new_row = {"Model": "RandomForestRegressor","MAE": mae, "MSE": mse, "R2 Score": r_squared, "R2 Score (Cross-Validation)": r2_cross_val}
models = models.append(new_row, ignore_index=True)

# %%
# Create a Decision Tree Regressor model
dtr = DecisionTreeRegressor()

# Compute the RMSE using cross-validation
r2_cv_score = r2_cv(dtr)

# Fit the model to the data and make predictions
dtr.fit(X_train, y_train)
y_pred = dtr.predict(X_test)

# Compute the evaluation metrics
mae, mse, r_squared = evaluation(y_test, y_pred)

print("MAE:", mae)
print("MSE:", mse)
print("R2 Score:", r_squared)
print("-"*30)
r2_cross_val = r2_cv(dtr)
print("R2 Scores(Cross-Validation):", r2_cross_val)

new_row = {"Model": "DecisionTree","MAE": mae, "MSE": mse, "R2 Score": r_squared, "R2 Score (Cross-Validation)": r2_cross_val}
models = models.append(new_row, ignore_index=True)

# %%

# Create a Linear Regression model as the base estimator for the Bagging model
lr = LinearRegression()

# Create a Bagging Regressor model with 10 estimators
bagging = BaggingRegressor(base_estimator=lr, n_estimators=10)

r2_cv_score = r2_cv(bagging)

# Fit the model to the data and make predictions
bagging.fit(X_train, y_train)
y_pred = bagging.predict(X_test)

# Compute the evaluation metrics
mae, mse, r_squared = evaluation(y_test, y_pred)

print("MAE:", mae)
print("MSE:", mse)
print("R2 Score:", r_squared)
print("-"*30)
r2_cross_val = r2_cv(bagging)
print("R2 Scores(Cross-Validation):", r2_cross_val)

new_row = {"Model": "Bagging","MAE": mae, "MSE": mse, "R2 Score": r_squared, "R2 Score (Cross-Validation)": r2_cross_val}
models = models.append(new_row, ignore_index=True)

# %%
# Create a Gradient Boosting Regressor model
gbr = GradientBoostingRegressor()

# Compute the RMSE using cross-validation
r2_cv_score = r2_cv(gbr)

# Fit the model to the data and make predictions
gbr.fit(X_train, y_train)
y_pred = gbr.predict(X_test)

# Compute the evaluation metrics
mae, mse, r_squared = evaluation(y_test, y_pred)

print("MAE:", mae)
print("MSE:", mse)
print("R2 Score:", r_squared)
print("-"*30)
r2_cross_val = r2_cv(gbr)
print("R2 Scores(Cross-Validation):", r2_cross_val)

new_row = {"Model": "GradientBoosting","MAE": mae, "MSE": mse, "R2 Score": r_squared, "R2 Score (Cross-Validation)": r2_cross_val}
models = models.append(new_row, ignore_index=True)

# %%
#XGBoost Regressor
xgb = XGBRegressor(n_estimators=1000, learning_rate=0.01)
xgb.fit(X_train, y_train)
predictions = xgb.predict(X_test)

mae, mse, r_squared = evaluation(y_test, predictions)
print("MAE:", mae)
print("MSE:", mse)
print("R2 Score:", r_squared)
print("-"*30)
r2_cross_val = r2_cv(xgb)
print("R2 Score (Cross-Validation)", r2_cross_val)

new_row = {"Model": "XGBRegressor","MAE": mae, "MSE": mse, "R2 Score": r_squared, "R2 Score (Cross-Validation)": r2_cross_val}
models = models.append(new_row, ignore_index=True)


# %%
#Support Vector Machines
svr = SVR(C=5)
svr.fit(X_train, y_train)
predictions = svr.predict(X_test)

mae, mse, r_squared = evaluation(y_test, predictions)
print("MAE:", mae)
print("MSE:", mse)
print("R2 Score:", r_squared)
print("-"*30)
r2_cross_val = r2_cv(svr)
print("R2 Score (Cross-Validation)", r2_cross_val)

new_row = {"Model": "SVR","MAE": mae, "MSE": mse, "R2 Score": r_squared, "R2 Score (Cross-Validation)": r2_cross_val}
models = models.append(new_row, ignore_index=True)

# %%
#Model Comparison 
models.sort_values(by="R2 Score (Cross-Validation)")

# %%
plt.figure(figsize=(12,8))
sns.barplot(x=models["Model"], y=models["R2 Score (Cross-Validation)"])
plt.title("Models' R2 Scores (Cross-Validated)", size=15)
plt.xticks(rotation=30, size=12)
plt.show()


