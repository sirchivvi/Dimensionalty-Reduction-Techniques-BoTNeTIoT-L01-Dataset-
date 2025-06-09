def compare_techniques(accuracies):
    """
    Compare the performance of different dimensionality reduction techniques.

    Parameters:
    accuracies (dict): Dictionary of technique names and their accuracy scores
    """
    print("\n=== Comparative Analysis ===")

    # Create comparison dataframe
    comparison_df = pd.DataFrame.from_dict(accuracies, orient='index', columns=['Accuracy'])
    comparison_df = comparison_df.sort_values('Accuracy', ascending=False)

    # Plot comparison
    plt.figure(figsize=(10, 6))
    sns.barplot(x=comparison_df.index, y=comparison_df['Accuracy'])
    plt.title('Model Accuracy by Dimensionality Reduction Technique')
    plt.ylabel('Accuracy')
    plt.xticks(rotation=45)
    plt.ylim(0, 1)
    plt.show()

    # Print insights
    print("\nKey Insights:")
    print("1. The best performing technique is:", comparison_df.idxmax()[0])
    print("2. The accuracy difference between best and worst is:",
          f"{comparison_df.max()[0] - comparison_df.min()[0]:.4f}")
    print("\nDetailed Comparison:")
    print(comparison_df)


    # Explain results
    print("\nExplanation of Results:")
    print("- LDA typically performs well when we have labeled data as it maximizes class separability.")
    print("- PCA is unsupervised and focuses on variance, which may not always align with classification goals.")
    print("- t-SNE is great for visualization but may not preserve global structure needed for classification.")
    print("- MDS preserves distances but can be computationally expensive for large datasets.")
    print("- SVD is similar to PCA but works directly with the data matrix rather than covariance matrix.")
