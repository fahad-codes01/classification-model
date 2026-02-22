# Classification Model Project

A portfolio-ready machine learning project demonstrating various classification algorithms on structured and text datasets.

## Project Overview

This project showcases machine learning classification techniques using scikit-learn. It includes implementations for both structured data (tabular) and text data classification tasks.

## Datasets Used

### Structured Data
- **Titanic Dataset**: Passenger survival prediction (binary classification)
- **Breast Cancer Dataset**: Breast cancer diagnosis classification (benign/malignant)

### Text Data
- **Pizza Reviews**: Thai text sentiment analysis
- **Burger King Reviews**: Thai text sentiment analysis

## Project Structure

```
Classification model/
├── data/
│   ├── raw/
│   │   ├── structured/       # Original CSV files
│   │   └── text/            # Original text datasets
│   └── processed/            # Cleaned/processed data
├── notebooks/
│   ├── 01_structured/
│   │   ├── titanic/         # Titanic classification experiments
│   │   └── breast_cancer/   # Breast cancer classification experiments
│   └── 02_text/
│       ├── pizza/           # Pizza review sentiment analysis
│       └── burgerking/     # Burger King review sentiment analysis
├── src/
│   ├── preprocessing.py     # Data preprocessing utilities
│   └── utils.py             # Model training and evaluation utilities
├── .gitignore
├── LICENSE
├── requirements.txt
└── README.md
```

## How to Run

1. **Clone the repository**
   
```
bash
   git clone <repository-url>
   
```

2. **Create a virtual environment**
   
```
bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   
```

3. **Install dependencies**
   
```
bash
   pip install -r requirements.txt
   
```

4. **Open and run notebooks**
   - Navigate to `notebooks/` directory
   - Open Jupyter notebooks with JupyterLab or Jupyter Notebook
   
```
bash
   jupyter lab
   
```

## Models Implemented

### Structured Data Classification
- K-Nearest Neighbors (KNN)
- Decision Tree
- Naive Bayes

### Text Classification
- Decision Tree with CountVectorizer
- TF-IDF vectorization
- Various NLP preprocessing techniques

## Results Summary

| Dataset | Model | Accuracy |
|---------|-------|----------|
| Titanic | KNN | ~80-85% |
| Breast Cancer | KNN | ~90-95% |
| Pizza Reviews | Decision Tree | Variable |
| Burger King Reviews | Decision Tree | Variable |

*Note: Results vary based on data preprocessing and model hyperparameters.*

## Future Improvements

- Add more classification algorithms (Random Forest, SVM, XGBoost)
- Implement hyperparameter tuning with GridSearchCV
- Add cross-validation for more robust evaluation
- Include data visualization for model interpretation
- Add feature engineering for improved performance

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Dataset sources: Kaggle and various open-source datasets
- Built with scikit-learn, pandas, and numpy
