import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import data_preprocessing as dp
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
import pickle
import shap

# import preprocessed data from csv file
path = r'C:\Users\Jairo\OneDrive\Documentos\ME_BreastCancer\data\processed\data_preprocessed.csv'

# split data into train and test sets
def split_data(path):
    df = pd.read_csv(path, encoding='utf-8-sig')
    X = df.drop('target', axis=1)
    #columns = X.columns
    y = df['target']
    print(X.shape)
    print(y.shape)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=101)
    return X_train, X_test, y_train, y_test

# scale data
def scale_data(X_train, X_test):
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    return X_train, X_test

# logistic regression
def logistic_regression(X_train, X_test, y_train, y_test):
    log_model = LogisticRegression()
    log_model.fit(X_train, y_train)
    predictions = log_model.predict(X_test)
    # save model
    model_path= 'C:/Users/Jairo/OneDrive/Documentos/ME_BreastCancer/models/log_model.pkl'
    directory = os.path.dirname(model_path)
    if not os.path.exists(directory):
        os.makedirs(directory)
    
    
    with open(model_path, 'wb') as file:
        pickle.dump(log_model, file)
    return predictions

# evaluate model with f1 score
def evaluate_model(y_test, predictions):
    print('Classification Report')
    print(classification_report(y_test, predictions))
    print('Confusion Matrix')
    print(confusion_matrix(y_test, predictions))
    print('Accuracy Score')
    print(accuracy_score(y_test, predictions))
    print('Precision Score')
    print(precision_score(y_test, predictions))
    print('Recall Score')
    print(recall_score(y_test, predictions))
    print('F1 Score')
    print(f1_score(y_test, predictions))
    # plot the confusion matrix
    cm = confusion_matrix(y_test, predictions)
    plt.figure(figsize=(10,10))
    sns.heatmap(cm, annot=True, fmt=".3f", linewidths=.5, square=True, cmap='Blues_r')
    plt.ylabel('Actual label')
    plt.xlabel('Predicted label')
    all_sample_title = 'Accuracy Score: {0}'.format(accuracy_score(y_test, predictions))
    plt.title(all_sample_title, size=15)
    plt.show()


# evaluate model with cross validation
def evaluate_model_with_cross_validation(X_train, y_train):
    log_model = LogisticRegression()
    scores = cross_val_score(log_model, X_train, y_train, cv=10, scoring='f1')
    print(scores)
    print('Mean F1 Score')
    print(scores.mean())

#Feature importance using SHAP
def feature_importance_shap(X_train, y_train, path):
    log_model = LogisticRegression()
    log_model.fit(X_train, y_train)
    explainer = shap.LinearExplainer(log_model, X_train)
    shap_values = explainer.shap_values(X_train)
    shap.summary_plot(shap_values, X_train, plot_type="bar")
    # import preprocessed data from csv file
    X = pd.read_csv(path, encoding='utf-8-sig')
    X = X.drop('target', axis=1)
    columns = X.columns


    # get names of the features
    feature_names = columns.values.tolist()
    # get the shap.summary_plot with the columns names
    shap.summary_plot(shap_values, X_train, plot_type="bar", feature_names=feature_names)
    

def coefficients(X_train, y_train, path):
    df = pd.read_csv(path, encoding='utf-8-sig')
    X = df.drop('target', axis=1)
    y = df['target']
    print(X.shape)
    print(y.shape)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=101)
    log_model = LogisticRegression()
    log_model.fit(X_train, y_train)
    coefficients = log_model.coef_
    coefficients = coefficients.tolist()
    coefficients = coefficients[0]
    coefficients = pd.DataFrame(coefficients)
    coefficients = coefficients.rename(columns={0: 'coefficients'})
    coefficients['features'] = X_train.columns
    coefficients = coefficients.sort_values(by='coefficients', ascending=False)
    coefficients = coefficients.reset_index(drop=True)
    # plot the coefficients
    plt.figure(figsize=(20,20))
    sns.barplot(x='coefficients', y='features', data=coefficients)
    plt.show()
    return coefficients
    

if __name__ == '__main__':
    X_train, X_test, y_train, y_test = split_data(path)
    X_train, X_test = scale_data(X_train, X_test)
    predictions = logistic_regression(X_train, X_test, y_train, y_test)
    evaluate_model(y_test, predictions)
    evaluate_model_with_cross_validation(X_train, y_train)
    feature_importance_shap(X_train, y_train, path)
    coefficients = coefficients(X_train, y_train, path)





