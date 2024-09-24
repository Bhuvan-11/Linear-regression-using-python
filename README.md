Task given: Predicting House Prices with Linear Regression

Code that i used for this task 
# Step 1: Installing Libraries (Skip if already installed)
!pip install pandas numpy scikit-learn matplotlib seaborn

# Step 2: Importing Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Step 3: Loading Dataset (Replace 'housing.csv' with your actual file)
df = pd.read_csv('D:\Jupyter notebook\Datasets\Housing.csv')

# Data preprocessing
df = pd.get_dummies(df, drop_first=True)

# Feature and target separation
X = df.drop('price', axis=1)
y = df['price']

# Step 6: Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 7: Model Training
model = LinearRegression()
model.fit(X_train, y_train)

# Step 8: Prediction and Evaluation
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f'Mean Squared Error: {mse}')
print(f'R-squared: {r2}')

# Step 9: Visualization
plt.scatter(y_test, y_pred, color='blue', label='Predicted Prices', marker='x')
plt.scatter(y_test, y_test, color='red', label='Actual Prices', marker='o')
plt.xlabel("Actual Prices")
plt.ylabel("Predicted Prices")
plt.title("Actual vs Predicted Prices")
plt.show()


Result

![image](https://github.com/user-attachments/assets/9905af09-361f-4e25-a768-0e40a7a05919)

