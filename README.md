# Logistic Regression Classifier from CSV

This project uses a logistic regression model to predict **Pass/Fail** outcomes based on study hours. It loads training data from a CSV file, trains a classifier, and prints predictions with confidence levels.

## 🔍 What It Does

- Loads data from `study_data_logistic.csv`
- Trains a logistic regression model using `scikit-learn`
- Predicts outcomes for new study hour values
- Prints result and confidence level for each prediction

## 📁 Files

- `csv_logistic_classifier.py` – Runs the model and prints results  
- `study_data_logistic.csv` – Input data used for training

## ⚙️ Requirements

- Python 3.12  
- `pandas`  
- `scikit-learn`  

Install with:

```bash
pip install pandas scikit-learn
