from sklearn.model_selection import cross_validate
import pandas as pd
import numpy as np


def cross_validation_func(model, X_train, y_train, c_v=5):
    scoring_array = ['accuracy', 'precision', 'recall', 'f1']
    results_scores = cross_validate(estimator=model,
                               X=X_train,
                               y=y_train,
                               cv=c_v,
                               scoring=scoring_array,
                               return_train_score=True)
      
    return {
              "Mean Train Accuracy": results_scores['train_accuracy'].mean(),
              "Mean Train Precision": results_scores['train_precision'].mean(),
              "Mean Train Recall": results_scores['train_recall'].mean(),
              "Mean Train F1 Score": results_scores['train_f1'].mean(),
              "Mean Valid Accuracy": results_scores['test_accuracy'].mean()*100,
              "Mean Valid Precision": results_scores['test_precision'].mean(),
              "Mean Valid Recall": results_scores['test_recall'].mean(),
              "Mean Valid F1 Score": results_scores['test_f1'].mean()
              }