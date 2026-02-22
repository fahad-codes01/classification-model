"""
Preprocessing utilities for machine learning pipelines.

This module provides functions for:
- Splitting data into train/test sets
- Cleaning structured data
- Cleaning text data
- Vectorizing text data
"""

import pandas as pd
import numpy as np
import re
import string
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer


def split_data(X, y, test_size=0.2, random_state=42):
    """
    Split data into training and testing sets.
    
    Parameters:
    -----------
    X : array-like, pandas DataFrame or numpy array
        Features
    y : array-like, pandas Series or numpy array
        Target variable
    test_size : float, optional (default=0.2)
        Proportion of data to use for testing
    random_state : int, optional (default=42)
        Random seed for reproducibility
        
    Returns:
    --------
    X_train, X_test, y_train, y_test : tuples
        Split data
    """
    return train_test_split(
        X, y, 
        test_size=test_size, 
        random_state=random_state,
        stratify=y if hasattr(y, 'shape') else None
    )


def basic_structured_cleaning(df):
    """
    Apply basic cleaning to structured data.
    
    Parameters:
    -----------
    df : pandas DataFrame
        Input dataframe
        
    Returns:
    --------
    df : pandas DataFrame
        Cleaned dataframe
    """
    df = df.copy()
    
    # Fill missing numeric values with median
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if df[col].isnull().any():
            df[col].fillna(df[col].median(), inplace=True)
    
    # Fill missing categorical values with mode
    categorical_cols = df.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        if df[col].isnull().any():
            df[col].fillna(df[col].mode()[0] if len(df[col].mode()) > 0 else 'Unknown', inplace=True)
    
    return df


def basic_text_cleaning(text_series):
    """
    Apply basic cleaning to text data.
    
    Parameters:
    -----------
    text_series : pandas Series
        Series containing text data
        
    Returns:
    --------
    cleaned_series : pandas Series
        Cleaned text series
    """
    def clean_text(text):
        if pd.isna(text):
            return ""
        
        text = str(text)
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove punctuation
        text = text.translate(str.maketrans('', '', string.punctuation))
        
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        return text
    
    return text_series.apply(clean_text)


def vectorize_text(train_text, test_text, method='count', max_features=300, ngram_range=(1, 3)):
    """
    Vectorize text data using CountVectorizer or TfidfVectorizer.
    
    Parameters:
    -----------
    train_text : array-like
        Training text data
    test_text : array-like
        Test text data
    method : str, optional (default='count')
        Vectorization method: 'count' for CountVectorizer, 'tfidf' for TfidfVectorizer
    max_features : int, optional (default=300)
        Maximum number of features
    ngram_range : tuple, optional (default=(1, 3))
        N-gram range
        
    Returns:
    --------
    X_train, X_test, vectorizer : tuple
        Vectorized data and fitted vectorizer
    """
    if method == 'tfidf':
        vectorizer = TfidfVectorizer(
            analyzer='word',
            ngram_range=ngram_range,
            max_features=max_features
        )
    else:
        vectorizer = CountVectorizer(
            analyzer='word',
            ngram_range=ngram_range,
            max_features=max_features
        )
    
    X_train = vectorizer.fit_transform(train_text)
    X_test = vectorizer.transform(test_text)
    
    return X_train, X_test, vectorizer
