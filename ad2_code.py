# Importing necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns

# Step 1: Load the dataset
# Assuming you have a CSV file named 'bank_customer_data.csv'
dataset = pd.read_csv('churn.csv')

# Step 2: Explore the dataset
print(dataset.head())
print(dataset.info())
print(dataset.describe())

# Step 3: Preprocessing

# Encoding categorical variables
label_encoder = LabelEncoder()

# Example: encoding gender, geography, etc.
dataset['Gender'] = label_encoder.fit_transform(dataset['Gender'])
dataset['Geography'] = label_encoder.fit_transform(dataset['Geography'])

# Define the features (X) and target (y)
features = dataset.drop(columns=['CustomerId', 'Surname', 'Exited'])  # Exclude non-informative columns
target = dataset['Exited']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

# Standardize the feature variables
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Step 4: Building the model
# Using Logistic Regression as the baseline model

log_reg = LogisticRegression()

# Hyperparameter tuning using GridSearchCV
param_grid = {
    'C': [0.1, 1, 10],
    'solver': ['liblinear', 'lbfgs', 'newton-cg'],
    'max_iter': [100, 200, 300]
}

grid_search = GridSearchCV(log_reg, param_grid, cv=5, n_jobs=-1, verbose=1)
grid_search.fit(X_train, y_train)

# Best model from GridSearchCV
best_model = grid_search.best_estimator_

# Step 5: Model Evaluation
# Predictions
y_pred = best_model.predict(X_test)
y_pred_proba = best_model.predict_proba(X_test)[:, 1]

# Performance metrics
print("Accuracy:", accuracy_score(y_test, y_pred))
print("AUC Score:", roc_auc_score(y_test, y_pred_proba))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Stayed', 'Exited'], yticklabels=['Stayed', 'Exited'])
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()

# Step 6: Predicting churn for a new customer
# Example: a new customer feature array with 11 features
new_customer = np.array([[40, 2, 60000, 1, 1, 1, 1, 0, 1, 0.0, 1]])  # Example data with 11 features

# Apply the same standardization as done during training
new_customer = scaler.transform(new_customer)  # Standardizing the new customer data

# Predicting churn for the new customer
churn_prediction = best_model.predict(new_customer)
churn_probability = best_model.predict_proba(new_customer)[:, 1]

print(f"Will the customer churn? {'Yes' if churn_prediction[0] == 1 else 'No'}")
print(f"Probability of churn: {churn_probability[0]:.2f}")
