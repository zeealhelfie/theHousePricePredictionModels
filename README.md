# House Price Prediction 🏡

**Modeling the Factors Influencing Real Estate Market.**

## Introduction

The "House Price Prediction 🏡" project aims to bridge the gap between data science and the real estate industry. Accurate house price prediction is of paramount importance, not only for homebuyers and sellers but also for real estate professionals, investors, and researchers. By developing a predictive model that leverages various features and factors, we provide a valuable tool for understanding the dynamic and complex world of housing prices.

In a market influenced by numerous variables, this project delves into the intricate interplay between factors that determine property values. Our dataset, consisting of 21,613 records in King County, Washington State, spans the period between May 2014 and May 2015. With 21 columns of data, this comprehensive collection offers a unique opportunity to uncover the nuances driving house prices.

This project delves into essential aspects of house price prediction, including data preprocessing, feature selection, and the application of diverse machine learning models. It also assesses the generalization capabilities of these models through 5-fold cross-validation. The project reveals insights that will empower stakeholders in the real estate market and contribute to more informed decision-making.

## Table of Contents

1. **Introduction**
   - Overview of the Project
   - Importance of House Price Prediction
   - Dataset Overview
   - Response vs. Predictor Variables

2. **Data Preprocessing**
   - Exploratory Data Analysis
   - Skewness Correction with Logarithmic Transformation
   - Data Correlation Analysis

3. **Feature Selection**
   - Stepwise Selection
   - Pearson Correlation
   - Selected Features

4. **Scaling and Data Split**
   - Scaling by Standardization
   - Data Split into Training and Testing Sets

5. **Selecting Performance Metrics**
   - Mean Absolute Error (MAE)
   - Mean Squared Error (MSE)
   - R-squared (R²) Score
   - R² Score with 5-Fold Cross-Validation

6. **Modeling**
   - Models and Hyperparameters
   - Model Training and Evaluation
   - Model Results

7. **Conclusion**
   - Model Performance
   - Generalization Ability
   - Promising Models

8. **Future Directions**
   - Areas for Further Research and Development

9. **References**
   - Citations and Resources


## Overview: The objective of this project
- Develop a predictive model that accurately estimates house prices based on various features and factors.
- Bridges the gap between data science and real estate, offering a valuable tool for industry professionals, researchers, and anyone interested in understanding the dynamics of housing prices.

## Why is this project important?

- Accurate house price prediction is crucial for both buyers and sellers in the real estate market to make informed decisions.
- Understanding the factors that influence house prices helps homeowners, real estate agents, and investors in setting competitive prices, identifying potential investment opportunities, and maximizing returns.

## Dataset Overview: 

The dataset used in this project contains house sale prices for King County,  Washington State.  It includes information about homes sold between May 2014 and May 2015. 
The dataset contains Rows: 21,613, Columns: 21.

<img width="354" alt="housingdataset" src="https://github.com/zeealhelfie/theHousePricePredictionModels/assets/60905286/01682934-b9f1-4aec-8857-84d621a6e474">

## Response vs. Predictor Variables:

- Response Variable: price
- Predictor Variables: 20
Id, sqft_living, bedrooms, bathrooms, sqft_lot, floors, waterfront, view, condition, grade, sqft_above, sqft_basement, yr_built, yr_renovated

## Closer look at the Response variable: price

- The histogram plot of the target variable “Price” shows that the distribution of the target variable is skewed to the right.

![histo](https://github.com/zeealhelfie/theHousePricePredictionModels/assets/60905286/b9de6975-5e84-4eab-91de-20cadeb0bc7e)

## Skewness correction: Adding a logarithmic transformation

- Applying a logarithmic transformation can help reduce the skewness and make the variable more normally distributed. 
- Logarithmic transformations can help stabilize the variance of the 'price' variable. If the variability of the variable increases or decreases as its magnitude changes, This transformation can make the variability more consistent.
- A logarithmic transformation can compress the scale of the data, which can help mitigate the influence of outliers and make the distribution more symmetrical.
![histo2](https://github.com/zeealhelfie/theHousePricePredictionModels/assets/60905286/6b81947c-099d-438e-a182-9a83b7926d88)

## Data Correlation: corrplot function

- Color gradients are used in correlation visualizations to represent the strength of the correlation.
- Blue represents a positive correlation, red represents a negative correlation, and shades in between represent varying strengths of the correlation.
- The intensity of the color corresponds to the magnitude of the correlation coefficient, with brighter or darker colors indicating stronger correlations.
  ![cor](https://github.com/zeealhelfie/theHousePricePredictionModels/assets/60905286/18bf516e-2462-46ed-a9ff-f2341e1d0103)

## Feature Selection:

Feature selection involves identifying the most relevant and informative features from the dataset to improve model performance and interpretability.
- Stepwise Selection: Both, systematically evaluate different combinations of features by adding or removing variables based on certain criteria.
- Pearson Correlation: measures the linear relationship between two numerical variables.

Stepwise Selection: 16 features
Based on the p-values in the summary,
we can determine which variables are statistically significant in relation to the response variable.

<img width="565" alt="step" src="https://github.com/zeealhelfie/theHousePricePredictionModels/assets/60905286/041553fc-a4ef-4e7c-91ee-1ff04c7fbd2b">

Pearson Correlation:

- Bathrooms (0.550802)
- sqft_living (0.695341)
- grade (0.703634)
- sqft_above (0.601801)

## Scaling by Standardization: StandardScaler is used for scaling

- Standardization scales the features so that they have a mean of 0 and a standard deviation of 1. 
- Scaling is primarily used to normalize the range and scale of the -features in a dataset. 
- Scaling by standardization can weaken the impact of outliers to some extent.

<img width="471" alt="scaler" src="https://github.com/zeealhelfie/theHousePricePredictionModels/assets/60905286/a9fb209d-9328-4a71-8db3-103a84d39ec0">

## ​​Data Split:

The dataset was split into training and testing: randomly partitioned the data, assigning 70% for training and the remaining 30% for testing.

## Selecting the performance metrics:

- Mean Absolute Error (MAE): How close the predictions to the the response variable.
- Mean Squared Error (MSE): MSE provides a measure of the average variance of the errors. 
- R-squared (R²): measures the variance in the response variable. overall measure of how well the model fits the data and captures the relationship between the X and Y variables.
- R-squared score was calculated using cross-validation: the number of cross-validation folds = 5.


## Predictions:

- model.fit(X_train, y_train)
- predictions = model.predict(X_test)
- mae, mse, r_squared = evaluation(y_test, predictions)


## Models and hyperparameters:


<img width="483" alt="models used in the project" src="https://github.com/zeealhelfie/theHousePricePredictionModels/assets/60905286/5e881b84-a218-42b3-bceb-6879a527e357">


## Results:

The results of our house price prediction models are summarized in the table above. Among the regression models, Lasso, ElasticNet, Linear Regression, Ridge, and Bagging demonstrated similar performance, with MAE values ranging from 0.199276 to 0.201649. These models achieved an R2 Score of approximately 0.76, indicating that they can explain around 76% of the variance in house prices. The Decision Tree model outperformed the regression models in terms of MAE (0.177384) and achieved an R2 Score of 0.776211. This suggests that decision trees effectively capture non-linear relationships and patterns in the data. Among the classification models, SVR, Gradient Boosting, Random Forest Regressor, and XGB Regressor showcased outstanding performance. These models achieved lower MAE and MSE values compared to the regression models. The XGB Regressor exhibited the best performance with an MAE of 0.121751, MSE of 0.028441, and an impressive R2 Score of 0.899278. Overall, the results indicate that the ensemble-based models, including Gradient Boosting, Random Forest Regressor, and XGB Regressor, provide the highest predictive accuracy for house price prediction. 

<img width="690" alt="models results" src="https://github.com/zeealhelfie/theHousePricePredictionModels/assets/60905286/154f7bda-70d4-4c5f-ba38-2ae5f1178b5e">

![r^2plot](https://github.com/zeealhelfie/theHousePricePredictionModels/assets/60905286/f7b5556f-6b43-4be9-8f15-69a9a067951c)


## Conclusion: 

In conclusion, the analysis demonstrates the effectiveness of various regression and classification models for predicting house prices based on a broad set of features. The ensemble models, specifically Gradient Boosting, Random Forest Regressor, and XGB Regressor, exceeded the other models, displaying lower MAE and MSE values and higher R2 Scores. The results of this analysis have important implications for stakeholders in the real estate industry, such as homebuyers, sellers, and investors. Accurate house price predictions enable informed decision-making and can assist in determining fair market values, identifying investment opportunities, and optimizing pricing strategies.

## Future Directions:

Proceeding ahead, there are several advances  for future research and improvement in house price prediction:

- Exploring additional relevant features.
- Investigating more advanced machine learning algorithms.
- Updating the Model As new data becomes available.
- Enhancing model interpretability.

By addressing these improvements, we can continue to enhance house price prediction models, delivering valuable tools for different industries and individuals.


