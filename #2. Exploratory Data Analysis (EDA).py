def perform_eda(df):
    """
    Perform exploratory data analysis on the dataset.

    Parameters:
    df (DataFrame): Processed dataset
    """
    print("\n=== EDA ===")

    # Class distribution
    plt.figure(figsize=(8, 6))
    sns.countplot(x='label', data=df)
    plt.title('Class Distribution')
    plt.show()

    # Correlation matrix
    plt.figure(figsize=(12, 10))
    corr = df.corr()
    sns.heatmap(corr, cmap='coolwarm', annot=False)
    plt.title('Feature Correlation Matrix')
    plt.show()

    # Summary statistics
    print("\nSummary Statistics:")
    print(df.describe())
