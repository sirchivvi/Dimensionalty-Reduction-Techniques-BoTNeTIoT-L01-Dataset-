# Import necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.manifold import TSNE, MDS
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.pipeline import make_pipeline


# Load the dataset (adjust the path if needed)
df = pd.read_csv('/content/BotNeTIoT-L01_label_NoDuplicates.csv')
# Basic info
print("Dataset Shape:", df.shape)
print("\nDataset Info:")
df.info()
# Display first few rows
df.head()


# Load the dataset
def load_data(filepath):
    """
    Load the IoT intrusion detection dataset from CSV file.

    Parameters:
    filepath (str): Path to the CSV file

    Returns:
    DataFrame: Loaded dataset
    """
    df = pd.read_csv(filepath)
    print(f"Dataset loaded with shape: {df.shape}")
    print("\nFirst 5 rows:")
    print(df.head())
    return df


# Preprocessing function
def preprocess_data(df):
    """
    Preprocess the dataset by handling missing values, encoding categorical variables,
    and scaling numerical features.

    Parameters:
    df (DataFrame): Raw dataset

    Returns:
    tuple: (X, y) where X is features and y is labels
    """
    # Handle missing values if any
    if df.isnull().sum().any():
        print("\nHandling missing values...")
        df = df.dropna()

    # Encode categorical labels
    le = LabelEncoder()
    df['label'] = le.fit_transform(df['label'])

    # Separate features and labels
    X = df.drop('label', axis=1)
    y = df['label']

    # Scale numerical features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled, y
