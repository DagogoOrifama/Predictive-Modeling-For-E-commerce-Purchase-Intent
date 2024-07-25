# Predictive Modeling For E-commerce Purchase Intent

This project involves the development and evaluation of machine learning models to predict user purchase behavior in an e-commerce setting using a highly imbalanced dataset. The study focuses on understanding user interactions and predicting purchase outcomes.

## Table of Contents
- [Introduction](#introduction)
- [Datasets](#datasets)
- [Data Exploration and Preparation](#data-exploration-and-preparation)
- [Model Design and Implementation](#model-design-and-implementation)
- [Result Analysis and Discussion](#result-analysis-and-discussion)
- [Ethical, Legal, and Professional Considerations](#ethical-legal-and-professional-considerations)
- [Conclusion](#conclusion)
- [References](#references)

## Introduction

Understanding user behavior is essential for improving user experience and maximizing business outcomes in the fast-changing e-commerce landscape. This study provides an in-depth analysis of a dataset containing 12,330 user sessions, with the objective of determining whether a user session ends with a purchase using machine learning algorithms. The algorithms explored include Decision Trees and XGBoost Classifiers.

## Datasets

The dataset used in this study reveals a class imbalance, with 84.5% of sessions classified as negative (no purchase) and 15.5% as positive (purchase). It includes 18 variables, comprising 17 independent variables (features) and one dependent (target) variable. The features include both numerical and categorical attributes, such as the number of pages viewed, time spent on pages, bounce rate, exit rate, page value, proximity to special days, operating system, browser, region, traffic type, visitor type, weekend visitation, and month of the year.

## Data Exploration and Preparation

1. **Initial Exploration**:
    - Viewed the top and bottom rows using `head()` and `tail()` functions.
    - Summarized data types and checked for missing values with `info()` and `isnull().sum()` functions.
    - Displayed statistical summaries using `describe()`.
    - Confirmed no missing values and visualized the class imbalance using `value_counts()`.

2. **Correlation Analysis**:
    - Conducted correlation analysis to identify features highly correlated with the target variable, Revenue. Features with low correlation were considered for elimination, but all features were used due to the small feature set.

## Model Design and Implementation

1. **Data Splitting**:
    - Split the data into features (X) and target (y).
    - Used `StratifiedShuffleSplit` to ensure representative class distribution in training and testing sets (80-20 split).

2. **Preprocessing**:
    - Designed a pipeline-integrated preprocessor using `ColumnTransformer`:
        - Applied `MinMaxScaler` to numerical features.
        - Used `OneHotEncoder` for categorical features.

3. **Model Development**:
    - Implemented Decision Trees and XGBoost Classifiers within scikit-learn pipelines.
    - Addressed class imbalance using SMOTE (Synthetic Minority Over-sampling Technique).
    - Created an `EvaluateModel` function to assess performance using accuracy, precision, recall, F1-score, and AUC.

4. **Hyperparameter Tuning**:
    - Introduced hyperparameter tuning to improve model performance.
    - Evaluated models with and without hyperparameter tuning, storing results in a DataFrame.

## Result Analysis and Discussion

- **Evaluation Metrics**:
    - Compared performance metrics for Decision Trees and XGBoost with and without hyperparameter tuning.
    - Noted significant improvement in recall for Decision Trees after tuning.
    - Highlighted the F1-Score as a reliable measure for imbalanced datasets.
    
- **Visualization**:
    - Plotted evaluation metrics to compare models.
    - Identified XGBoost as the best-performing algorithm, balancing recall and precision effectively.

- **Confusion Matrix**:
    - Evaluated for the best model (XGBoost), showing accurate predictions for both classes.

## Ethical, Legal, and Professional Considerations

- Ensured the dataset did not contain biases that could affect prediction accuracy.
- Complied with data privacy regulations (GDPR, CCPA) and ensured the data had a free license for use.
- Documented the data mining and machine learning processes for transparency.

## Conclusion

This study aimed to forecast e-commerce transactions using data mining and machine learning techniques. The preprocessing steps, including one-hot encoding, feature scaling, and SMOTE oversampling, laid the foundation for model development. XGBoost outperformed Decision Trees in prediction accuracy and resilience, even with hyperparameter tuning. The comprehensive evaluation and F1-Score analysis support the deployment of XGBoost for predicting e-commerce purchase behavior.

## References
- Dairu, X., & Shilong, Z. (2021). Machine Learning Model for Sales Forecasting by Using XGBoost. IEEE International Conference on Consumer Electronics and Computer Engineering.
- Felipe Farias, T. L.-F. (2020). Similarity Based Stratified Splitting: an approach to train better classifiers. Cornell University.
- Ileberi, E., Sun, Y., & Wang, Z. (2021). Performance Evaluation of Machine Learning Methods for Credit Card Fraud Detection Using SMOTE and AdaBoost. IEEE Access.
- Md Manjurul Ahsan, A. (2021). Effect of Data Scaling Methods on Machine Learning Algorithms and Model Performance. MDPI.
- Patel, D., Shrivastava, S., Gifford, W., & Siegel, S. (2020). Smart-ML: A System for Machine Learning Model Exploration using Pipeline Graph. IEEE International Conference on Big Data.
- Peng, H., Huang, S., Geng, T., Li, A., & Jiang, W. (2021). Accelerating Transformer-based Deep Learning Models on FPGAs using Column Balanced Block Pruning. ISQED.
- Safavian, S., & Landgrebe, D. (1991). A survey of decision tree classifier methodology. IEEE Transactions on Systems, Man, and Cybernetics.
- Sakar, C., & Yumusak, N. (2018). Online Shoppers Purchasing Intention Dataset. UCI Machine Learning Repository.
- Sandha, S. S. (2022). Enabling Hyperparameter Tuning of Machine Learning Classifiers in Production. IEEE International Conference on Cognitive Machine Intelligence.
- Sinthong, P., & Carey, M. J. (2019). AFrame: Extending DataFrames for Large-Scale Modern Data Analysis. IEEE International Conference on Big Data.
