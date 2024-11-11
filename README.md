# Breast Cancer Wisconsin Diagnostic - Logistic Regression Analysis

This repository provides an analysis of the Breast Cancer Wisconsin Diagnostic dataset using Logistic Regression with L1 and L2 regularization. The goal is to classify tumor samples as malignant or benign based on various features, optimizing for accuracy with regularization techniques. The analysis also compares feature selection effectiveness and accuracy between L1 (Lasso) and L2 (Ridge) regularized logistic regression models.

## Dataset

The Breast Cancer Wisconsin Diagnostic dataset from the UCI Machine Learning Repository is used in this analysis. This dataset contains 30 features describing characteristics of cell nuclei from digitized images of fine needle aspirate (FNA) biopsies.

## Requirements

This code requires Python 3.x and the following libraries:
	•	numpy
	•	pandas
	•	matplotlib
	•	scikit-learn
	•	ucimlrepo (for fetching the dataset)

## Code Overview

### Data Loading

The fetch_ucirepo function from the ucimlrepo library is used to retrieve the dataset, split into features X and targets y.

### Data Splitting

The dataset is split into training and test sets using an 80/20 split.

### Logistic Regression Model (Baseline)

A baseline logistic regression model without regularization is trained and evaluated. This model achieves an accuracy score on the test data.

### Regularized Logistic Regression with Cross-Validation

To improve performance, Logistic Regression models with L1 (Lasso) and L2 (Ridge) regularization are implemented using LogisticRegressionCV with 5-fold cross-validation and varying regularization strengths (C values):
	•	L1 Penalty: Uses liblinear solver.
	•	L2 Penalty: Uses lbfgs solver.

### Results

	•	Optimal Regularization Parameter (C): The best C values for L1 and L2 penalties are found.
	•	Accuracy: Test set accuracy for both models.
	•	Feature Selection: The L1 model is more selective, using fewer non-zero coefficients, while L2 includes all features.

### Key Findings

	•	Accuracy: L1 penalty achieved a higher test accuracy (0.95) compared to L2 (0.93).
	•	Feature Selection: The L1 model used 18 features, while the L2 model used all 30 features, suggesting that L1 is more effective for feature selection.
	•	Regularization Strength: Optimal C values for L1 and L2 penalties were 21.54 and 1291.55, respectively.
