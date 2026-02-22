"""
Utility functions for machine learning model training and evaluation.

This module provides functions for:
- Training models
- Evaluating models
- Printing classification reports
"""

from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


def train_model(model, X_train, y_train):
    """
    Train a machine learning model.
    
    Parameters:
    -----------
    model : sklearn estimator
        The model to train
    X_train : array-like
        Training features
    y_train : array-like
        Training labels
        
    Returns:
    --------
    model : sklearn estimator
        Trained model
    """
    model.fit(X_train, y_train)
    return model


def evaluate_model(model, X_test, y_test):
    """
    Evaluate a trained model.
    
    Parameters:
    -----------
    model : sklearn estimator
        Trained model
    X_test : array-like
        Test features
    y_test : array-like
        Test labels
        
    Returns:
    --------
    metrics : dict
        Dictionary containing evaluation metrics
    """
    y_pred = model.predict(X_test)
    
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'confusion_matrix': confusion_matrix(y_test, y_pred),
        'classification_report': classification_report(y_test, y_pred)
    }
    
    return metrics


def print_classification_report(y_true, y_pred, target_names=None):
    """
    Print classification report.
    
    Parameters:
    -----------
    y_true : array-like
        True labels
    y_pred : array-like
        Predicted labels
    target_names : list, optional
        Names of classes
    """
    if target_names is None:
        print(classification_report(y_true, y_pred))
    else:
        print(classification_report(y_true, y_pred, target_names=target_names))
