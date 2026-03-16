# Customer Churn Prediction System
EDA → Model Training → Pipeline → FastAPI → Prediction API

## Project Overview
This project builds a machine learning system to predict customer churn using customer behavior data. The model identifies customers likely to leave based on usage patterns, payment behavior, and support interactions.

## Dataset
Synthetic churn dataset with ~64K customer records.

Key features include:
- Age
- Tenure
- Payment Delay
- Support Calls
- Monthly Charges

## Project Workflow
1. Exploratory Data Analysis (EDA)
2. Feature correlation analysis
3. Model training and evaluation
4. ML pipeline creation
5. Model serialization
6. FastAPI deployment for predictions

## Model Training
Models evaluated:
- Logistic Regression
- Random Forest

Cross-validation was used to ensure model robustness.

## API Deployment
A FastAPI service was built to serve predictions via REST API.

Example endpoint:

POST /predict

Returns churn prediction for a customer input.

## Tech Stack
Python  
Pandas  
Scikit-learn  
FastAPI  
Uvicorn  

## How to Run the Project

Clone the repository
