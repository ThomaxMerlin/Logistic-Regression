# Logistic-Regression

Car Purchase Prediction using Logistic Regression
With Jupyter Notebook

This Jupyter Notebook demonstrates how to build a Logistic Regression model to predict whether a customer will purchase a car based on features like Gender, Age, and AnnualSalary. The dataset used is car_data.csv, which contains 1000 entries with numerical and categorical features.

Table of Contents
Prerequisites

Getting Started

Running the Code

Code Explanation

Results

License

Prerequisites
Before running the code, ensure you have the following installed:

Python 3.x

Required Python libraries:

bash
Copy
pip install numpy pandas seaborn matplotlib scikit-learn statsmodels jupyter
Jupyter Notebook (to run the .ipynb file).

Getting Started
Download the Dataset
Ensure the dataset car_data.csv is in the same directory as the notebook.

Launch Jupyter Notebook
Start Jupyter Notebook:

bash
Copy
jupyter notebook
Open the .ipynb file from the Jupyter Notebook interface.

Running the Code
Open the .ipynb file in Jupyter Notebook.

Run each cell sequentially to execute the code.

Code Explanation
1. Import Libraries
python
Copy
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline
Libraries used for data manipulation, visualization, and modeling.

2. Load and Explore Data
python
Copy
data = pd.read_csv('car_data.csv')
data.head()
data.info()
data.isnull().sum()
Load the dataset and explore its structure, summary statistics, and check for missing values.

3. Data Preprocessing
python
Copy
data = data.drop(columns="User ID")
data = data.replace('Male', 1)
data = data.replace('Female', 0)
Drop the User ID column as it is not relevant for prediction.

Encode the Gender column into numerical values (1 for Male, 0 for Female).

4. Data Visualization
python
Copy
sns.pairplot(data, hue='Purchased')
sns.countplot(data=data, x='Purchased')
sns.countplot(data=data, x='Purchased', hue="Gender")
sns.countplot(data=data, x='Age', hue="Purchased")
Visualize relationships between features and the target variable (Purchased).

5. Correlation Analysis
python
Copy
data.corr()
Analyze correlations between features.

6. Variance Inflation Factor (VIF)
python
Copy
from statsmodels.stats.outliers_influence import variance_inflation_factor
x = data.drop(columns='Purchased')
vif_data = pd.DataFrame()
vif_data['feature'] = x.columns
vif_data["VIF"] = [variance_inflation_factor(x.values, i) for i in range(len(x.columns))]
print(vif_data)
Check for multicollinearity using VIF.

7. Split Input and Output
python
Copy
x = data.drop(columns="Purchased")
y = data['Purchased']
Separate the features (x) and target variable (y).

8. Standardize Features
python
Copy
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
x_scaled = scaler.fit_transform(x)
data_scaled = pd.DataFrame(x_scaled)
data_scaled.head()
Standardize the features for better model performance.

9. Build and Train Model
python
Copy
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
model = LogisticRegression()
model.fit(x_train, y_train)
Split the data into training and testing sets.

Train a Logistic Regression model on the training data.

10. Evaluate Model
python
Copy
y_pred = model.predict(x_test)
from sklearn.metrics import accuracy_score, confusion_matrix
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
Evaluate the model using accuracy and confusion matrix.

Results
Accuracy: The model's accuracy on the test set.

Confusion Matrix: A matrix showing the true vs predicted values.

Visualizations: Insights into the dataset and model performance.

License
This project is open-source and available under the MIT License. Feel free to use, modify, and distribute it as needed.

Support
If you encounter any issues or have questions, feel free to open an issue in this repository or contact me at minthukywe2020@gmail.com.

Enjoy exploring car purchase prediction using Logistic Regression in Jupyter Notebook! 🚀
