import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
# Loading dataset

data = pd.read_csv("/kaggle/input/advertisingcsv/Advertising.csv")
# Get the dimensions of the DataFrame

rows, columns = data.shape

# Create a text annotation to display the dimensions
plt.figure(figsize=(6, 4))
plt.text(0.5, 0.5, f'Rows: {rows}\nColumns: {columns}', fontsize=12, ha='center', va='center')
plt.axis('off')
plt.title('DataFrame Dimensions')
plt.show()
# Display the first 10 rows of the dataset

data.head(10)
# Drop the column
data.drop('Unnamed: 0', axis=1, inplace=True)
# Basic dataset information

data.info()
# Check for missing values

data.isnull().sum
# Descriptive statistics of the dataset

data.describe()
# Distribution of Sales

plt.figure(figsize=(10, 6))
sns.histplot(data['Sales'], bins=30, kde=True)
plt.title('Distribution of Sales')
plt.xlabel('Sales')
plt.ylabel('Frequency')
plt.show()
# Pairplot to visualize relationships between numerical features

sns.pairplot(data, vars=['TV', 'Radio', 'Newspaper', 'Sales'])
plt.title('Pairplot of Numerical Features ')
plt.show()
# Correlation heatmap

plt.figure(figsize=(8, 6))
correlation_matrix = data[['TV', 'Radio', 'Newspaper', 'Sales']].corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Heatmap')
plt.show()
# Create subplots for each histogram

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# Plot histograms for 'TV,' 'Radio,' and 'Newspaper' columns
data["TV"].plot.hist(ax=axes[0], bins=10, color='skyblue', edgecolor='black')
axes[0].set_title('TV Advertising Budget')
axes[0].set_xlabel('Spending')
axes[0].set_ylabel('Frequency')

data["Radio"].plot.hist(ax=axes[1], bins=10, color='lightcoral', edgecolor='black')
axes[1].set_title('Radio Advertising Budget')
axes[1].set_xlabel('Spending')
axes[1].set_ylabel('Frequency')

data["Newspaper"].plot.hist(ax=axes[2], bins=10, color='lightgreen', edgecolor='black')
axes[2].set_title('Newspaper Advertising Budget')
axes[2].set_xlabel('Spending')
axes[2].set_ylabel('Frequency')

plt.tight_layout()
plt.show()
# Split the data into training and testing sets

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(data[["TV"]], data[["Sales"]], test_size=0.3, random_state=42)
# Create a linear regression model

from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(x_train,y_train)
# Make predictions on the test data

predictions = model.predict(x_test)
# Plot the regression line

plt.figure(figsize=(8, 6))
plt.scatter(x_test, y_test, color='blue', label='Actual Sales')
plt.plot(x_test, predictions, color='red', linewidth=2, label='Predicted Sales')
plt.title('Linear Regression - TV Advertising vs. Sales')
plt.xlabel('TV Advertising Budget')
plt.ylabel('Sales')
plt.legend()
plt.show()