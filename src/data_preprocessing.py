import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
import numpy as np


path = r'C:\Users\Jairo\OneDrive\Documentos\ME_BreastCancer\data\raw\data.csv'

def preprocessing(path):
    
    # import data from csv
    def import_data(path):
        df = pd.read_csv(path)
        return df

    # drop id column
    def drop_id(df):
        df = df.drop('id', axis=1)
        return df

    # drop unnamed column
    def drop_unnamed(df):
        df = df.drop('Unnamed: 32', axis=1)
        return df

    # rename diagnosis column and replace 'M' and 'B' with 1 and 0
    def rename_diagnosis(df):
        df = df.rename(columns={'diagnosis': 'target'})
        df['target'] = df['target'].replace({'M': 1, 'B': 0})
        return df

    # check for missing values
    def check_missing_values(df):
        df.isnull().sum()
        return df

    # check for duplicates
    def check_duplicates(df):    
        df.duplicated().sum()
        return df

    # treating outliers with IQR
    def treat_outliers_with_iqr(df):
        # Calculate Q1, Q2 and IQR
        for col in df.columns:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
        
            # Define bounds
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
        
            # Treat outliers
            # Here, we are replacing the outliers with NaN. 
            # You can replace NaN with mean or median based on the requirement.
            df[col] = df[col].apply(lambda x: x if lower_bound < x < upper_bound else None)

            return df
        
    # feature selection
    def feature_selection(df):
        # Correlation matrix
        corr = df.corr()
        plt.figure(figsize=(20,20))
        sns.heatmap(corr, annot=True, fmt='.2f', cmap='Blues')
        plt.show()
        
        # Select upper triangle of correlation matrix
        upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
        
        # Find index of feature columns with correlation greater than 0.95
        to_drop = [column for column in upper.columns if any(upper[column] > 0.95)]
        
        # Drop features 
        df = df.drop(df[to_drop], axis=1)
        
        return df

    # save the preprocessed data to a csv file

    def save_preprocessed_data(df):
        data_preprocessed = df.to_csv(r'C:\Users\Jairo\OneDrive\Documentos\ME_BreastCancer\data\processed\data_preprocessed.csv', index=False)
        return data_preprocessed

    # call all functions
    df = import_data(path)
    df = drop_id(df)
    df = drop_unnamed(df)
    df = rename_diagnosis(df)
    df = check_missing_values(df)
    df = check_duplicates(df)
    df = treat_outliers_with_iqr(df)
    df = feature_selection(df)
    df = save_preprocessed_data(df)

if __name__ == '__main__':
    preprocessing(path)
