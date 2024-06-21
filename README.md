# Loan-Status-Prediction
Project Overview
This project aims to predict the loan status (approved or rejected) of applicants using a dataset from Kaggle. The dataset includes various features such as applicant income, loan amount, credit history, and more. By leveraging Support Vector Machine (SVM), a powerful classification algorithm, we aim to develop a robust predictive model.

Objectives
Data Preprocessing: Clean and preprocess the data to handle missing values, encode categorical variables, and normalize numerical features.
Feature Selection: Identify the most significant features that influence the loan status.
Model Development: Implement and train an SVM classifier on the preprocessed data.
Model Evaluation: Evaluate the performance of the SVM model using appropriate metrics such as accuracy, precision, recall, and F1-score.
Hyperparameter Tuning: Optimize the SVM model by tuning its hyperparameters to achieve the best predictive performance.
Model Deployment: Deploy the final model for real-time prediction of loan status.

Dataset
Source: Kaggle
Description: The dataset contains information about loan applicants, including demographic details, loan amount, loan term, credit history, and loan status.
Features:
Loan_ID: Unique identifier for each loan
Gender: Gender of the applicant
Married: Marital status of the applicant
Dependents: Number of dependents
Education: Education level of the applicant
Self_Employed: Self-employment status
ApplicantIncome: Income of the applicant
CoapplicantIncome: Income of the co-applicant
LoanAmount: Loan amount requested
Loan_Amount_Term: Term of the loan
Credit_History: Credit history of the applicant
Property_Area: Urban, semi-urban, or rural area
Loan_Status: Loan approval status (target variable)
Methodology
Data Collection: Import the dataset from Kaggle.
Data Exploration: Perform exploratory data analysis (EDA) to understand the distribution and relationships of the features.
Data Cleaning: Handle missing values through imputation or removal, and ensure data consistency.

Data Preprocessing:
Encode categorical variables using techniques such as one-hot encoding or label encoding.
Normalize numerical features to ensure they have a uniform scale.
Feature Engineering: Create new features or modify existing ones to improve model performance.

Model Training:
Split the dataset into training and testing sets.
Train an SVM classifier using the training set.

Tools and Technologies
Programming Language: Python

Libraries:
Data Manipulation: Pandas, NumPy
Data Visualization: Seaborn
Machine Learning: Scikit-learn

Results
The project successfully developed an SVM model capable of predicting loan status with high accuracy. The model's performance metrics demonstrated its effectiveness, and the hyperparameter tuning process further enhanced its predictive capability. The deployed model provides a user-friendly interface for real-time loan status prediction.

Conclusion
By leveraging SVM and a comprehensive data preprocessing pipeline, this project showcases the potential of machine learning in financial decision-making processes. The model's ability to accurately predict loan status can aid financial institutions in making informed lending decisions, thereby reducing risk and improving efficiency.

Future Work
Model Improvement: Explore advanced techniques such as ensemble learning to further boost model performance.
Feature Expansion: Incorporate additional features such as applicant's employment history or detailed credit scores.
Deployment: Develop a fully functional web application for broader accessibility and real-time usage.




