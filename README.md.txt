Project Objective :
The objective of this project is to use historical loan application data to predict whether or not an applicant will be able to repay a loan. The goal is to build a model that can identify the loan applicants who are likely to repay the loan, allowing companies to avoid losses and incur profits.

Introduction
An existential problem for any loan providers today is to find out the loan applicants who are very likely to repay the loan. This way companies can avoid losses and incur huge profits. Home Credit offers easy, simple and fast loans for a range of home appliances, mobile phones, laptops, two-wheelers, and varied personal needs. Home Credit wants us todevelop a model to find out the loan applicants who are capable of repaying a loan, given the applicant data, all credits data from Credit Bureau.

Problem Statement
Home Credit, a company that offers easy, simple and fast loans for a range of Home Appliances, Mobile Phones, Laptops, Two Wheeler's, and varied personal needs, wants us to develop a model to find out the loan applicants who are capable of repaying a loan, given the applicant data, all credits data from Credit Bureau.

Dataset Description
The brief description of datasets used in this project are given below:

applications_base.csv:
This is the main table, broken into two files for Train (with TARGET) and Test (without TARGET). Static data for all applications. One row represents one loan in the data sample.

bureau.csv:
All client's previous credits provided by other financial institutions that were reported to Credit Bureau (for clients who have a loan in the sample). For every loan in the sample, there are as many rows as number of credits the client had in Credit Bureau before the application date.


Requirements:
1. Python 3.x
2. Jupyter Notebook
3. Pandas
4. NumPy
5. Matplotlib
6. Seaborn
7. Sklearn
8. LightGBM
The above listed libraries and packages are necessary to run the project. Make sure to have them installed in your system before running the project.

Jupyter Notebook is used to run the code and visualize the results.

Pandas is used for data manipulation and analysis.

NumPy is used for numerical operations.

Matplotlib and Seaborn are used for data visualization.

Sklearn (Scikit-learn) is used for Machine Learning models and evaluation metrics.

LightGBM is used for training and prediction of the model.

Please use the latest version of the above listed libraries and packages for better compatibility with the code.

Methodology
The project is broken down into the following steps:

1.Exploratory Data Analysis (EDA)
2.Feature Engineering
3.Data Preparation
4.Machine Learning Modelling
5.Performance Metrics
6.Conclusion
The models used in this project are Light Gradient Boosting Algorithm (LGBM) and Logistic Regression

Results
The results of the project showed that

Project Title: Home Credit Risk Management Project Description: The objective of this project is to use historical loan application data to predict whether or not an applicant will be able to repay a loan. Home Credit, a company that offers easy, simple and fast loans for a range of Home Appliances, Mobile Phones, Laptops, Two Wheelers, and varied personal needs, wants us to develpo th emodel to find out the loan applicants who are capable of repaying a loan, given the applicant data, all credits data from Credit Bureau. The goal is to build a model to predict how capable each applicant is of repaying a loan, so that sanctioning loan only for the applicants who are likely to repay the loan.

Data: The dataset consists of several different types of data, including information about the applicant, their credit history.

Requirements: The project is built on python3 and requires the following packages:

numpy pandas seaborn matplotlib sklearn lightgbm scikitplot Installation: Clone the repository Create a virtual environment Install the required packages using pip Copy code pip install -r requirements.txt Usage: Run the jupyter notebook Run all the cells Results: The model uses Light Gradient Boosting Machine (LGBM) algorithm and compares it with Logistic Regression algorithm. The performance of the model is evaluated using various metrics such as Confusion matrix, Precision, Recall, F1 Score and ROC. The model with LGBM has a higher recall score of 64.59% and AUC of 0.73, compared to Logistic Regression model.

Further Improvements: Further tuning of the LGBM model's hyperparameters Ensemble techniques like XGboost and Random Forest Use of oversampling techniques like SMOTE to handle class imbalance. References: https://www.kaggle.com/c/home-credit-default-risk/overview/evaluation https://lightgbm.readthedocs.io/en/latest/ 

Project Steps:

Data Exploration: Understanding the structure and distribution of the data through visualizations and statistical analysis.

Feature Engineering: Transforming and creating new features from the existing data to improve model performance. Techniques such as data normalization, one-hot encoding, and handling imbalanced data were used.

Data Preparation: Cleaning and pre-processing the data for model building. This step includes handling missing values, outlier detection, and feature scaling.

Machine Learning Modelling: Developing predictive models using Light gradient boosting algorithm(LGBM) and logistic regression.

Performance Metrics: Evaluating the performance of the models using metrics such as confusion matrix, precision score, recall score, F1 score and AUC.

Conclusion: Summarizing the findings and discussing the limitations of the project.