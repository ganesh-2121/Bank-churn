Customer Churn Prediction - Bank Dataset


📋 Project Overview


This project focuses on predicting customer attrition (churn) for a bank using machine learning techniques.

We aim to identify customers likely to leave the bank, allowing proactive measures to improve customer retention.

📚 Dataset Description


The dataset contains various features about customers, such as:


CreditScore

Geography

Gender

Age

Tenure

Balance

NumOfProducts

HasCrCard

IsActiveMember

EstimatedSalary

Exited (Target variable: 1 = customer exited, 0 = customer stayed)

🛠️ Technologies Used


Python

Pandas

NumPy

Scikit-learn

Matplotlib

Seaborn

Jupyter Notebook

📈 Project Workflow


Data Preprocessing

Dropped unnecessary columns (RowNumber, CustomerId, Surname, etc.).

Mapped categorical columns into more readable string labels.

Applied standard scaling to features.

Exploratory Data Analysis (EDA)

Visualized distributions.

Used Sankey Diagrams to visualize customer behavior patterns (Has Credit Card, Is Active Member, etc.).

Feature Engineering

Created string versions of numeric columns for plotting.

Handled categorical variables properly.

Model Building

Trained Machine Learning models (like Logistic Regression, Random Forest, etc.).

Evaluated using metrics such as Accuracy, Precision, Recall, and F1-Score.

Deployment Suggestions

Model can be deployed as a REST API or integrated into internal bank software to predict customer churn live.

📊 Visualizations


Sankey Diagrams to show flow between Credit Card Status / Activity Status and Exited label.

Bar plots, Histograms, and Correlation Matrix for feature understanding.


📌 Project Conclusion


Identified key features influencing customer churn.

Built a predictive model that can help banks retain valuable customers.

Visualized customer behavior for better strategic decision-making.

✨ Future Enhancements
Hyperparameter tuning to improve model accuracy.

Deploying the model via Flask or FastAPI.

Create an interactive dashboard using Streamlit or PowerBI.
