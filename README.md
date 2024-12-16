# Data Science Course Homework

This project was part of my homework for the Data Science course by Akademia Data Science. The objective was to apply machine learning techniques to solve a problem using a real-world dataset. The project involved data preprocessing, addressing class imbalance, implementing multiple machine learning models, and evaluating their performance. Metrics such as accuracy, recall, precision, F1 score, and ROC-AUC were used to assess the effectiveness of the models. 

## Dataset
Credit Fraud dataset from Kaggle. 

## Context
This project aims to build a machine learning model that predicts the likelihood of a credit card applicant defaulting on payments. The model uses a variety of features, including demographic and financial information, to assess the risk involved in approving or denying a credit card application.

## Key Tasks
- **Data Preprocessing**: Handled missing data and applied feature scaling to prepare the dataset for model training.
- **Addressing class imbalance**: Used SMOTE to balance out the difference in counts.
- **Model Selection**: Trained various machine learning models, including DecisionTreeClassifier, Random Forest, and XGBClassifier.
- **K-Fold Cross-Validation**: Implemented K-Fold cross-validation to ensure robust performance of the models and reduce overfitting.
- **Model Evaluation**: Evaluated each model based on multiple metrics, including accuracy, recall, precision, F1 score, and ROC-AUC.
- **Best Model**: After evaluating the models, Random Forest achieved the highest performance and was selected as the best model.
  
## Results
- **Best Model**: Random Forest
- **Accuracy**: 99.52%
- **Recall**: 99.30%
- **Precision**: 99.74%
- **F1 Score**: 99.52%

## Dependencies
- Python 3.x
- Pandas
- NumPy
- Scikit-learn
- XGBoost
