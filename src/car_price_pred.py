import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn import metrics
import os
import subprocess

# Load and preprocess data
car_data = pd.read_csv(r"D:\internship\pro1\car_data.csv")
print(car_data.head(), car_data.info(), car_data.isnull().sum(), car_data.describe())
print(car_data['Fuel_Type'].value_counts(), car_data['Seller_Type'].value_counts(), car_data['Transmission'].value_counts())

# Visualize categorical data
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle('Categorical Data Visualization')
for ax, col, color in zip(axes, ['Fuel_Type', 'Seller_Type', 'Transmission'], ['royalblue', 'red', 'purple']):
    ax.bar(car_data[col], car_data['Selling_Price'], color=color)
    ax.set_xlabel(col)
axes[0].set_ylabel('Selling Price')
plt.show()

# Seaborn visualization
fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)
fig.suptitle('Categorical Columns with Seaborn')
for ax, col in zip(axes, ['Fuel_Type', 'Seller_Type', 'Transmission']):
    sns.barplot(x=col, y='Selling_Price', data=car_data, ax=ax)
plt.show()

# Group analysis
print(car_data.groupby('Fuel_Type').get_group('Petrol').describe())
print(car_data.groupby('Seller_Type').get_group('Dealer').describe())

# Encoding and correlation heatmap
car_data.replace({'Fuel_Type': {'Petrol': 0, 'Diesel': 1, 'CNG': 2}}, inplace=True)
car_data = pd.get_dummies(car_data, columns=['Seller_Type', 'Transmission'], drop_first=True)
sns.heatmap(car_data.select_dtypes(include=['int64', 'float64']).corr(), annot=True, cmap='coolwarm')
plt.title('Correlation Heatmap')
plt.show()

# Regression plot
sns.regplot(x='Present_Price', y='Selling_Price', data=car_data)
plt.title('Present Price vs Selling Price')
plt.show()

# Model training and evaluation
X, y = car_data.drop(['Car_Name', 'Selling_Price'], axis=1), car_data['Selling_Price']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
scaler = StandardScaler()
X_train, X_test = scaler.fit_transform(X_train), scaler.transform(X_test)

model = LinearRegression().fit(X_train, y_train)
pred = model.predict(X_test)

print(f"MAE: {metrics.mean_absolute_error(pred, y_test):.2f}", 
      f"MSE: {metrics.mean_squared_error(pred, y_test):.2f}", 
      f"R2 Score: {metrics.r2_score(pred, y_test):.2f}", sep='\n')

sns.regplot(x=pred, y=y_test)
plt.xlabel('Predicted Price')
plt.ylabel('Actual Price')
plt.title('Actual vs Predicted Price')
plt.show()

# Run subprocess
subprocess.run(['python', 'D:\\internship\\pro1\\tinker_based.py'])
