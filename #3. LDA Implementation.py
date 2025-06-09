def apply_lda(X, y):
    """
    Apply Linear Discriminant Analysis (LDA) to the dataset.

    Parameters:
    X (array): Scaled features
    y (array): Labels

    Returns:
    array: Transformed features
    """
    print("\n=== LDA ===")
    lda = LDA()
    X_lda = lda.fit_transform(X, y)

    # Plot LDA components (only 1D for binary classification)
    plt.figure(figsize=(8, 6))

    if X_lda.shape[1] == 1:
        # For 1 component, use a histogram or strip plot
        plt.scatter(X_lda[:, 0], np.zeros_like(X_lda[:, 0]), c=y, cmap='viridis', alpha=0.6)
        plt.xlabel('LDA Component 1')
        plt.yticks([])  # Hide y-axis as it's just for visualization
    else:
        # For 2+ components, use regular scatter plot
        scatter = plt.scatter(X_lda[:, 0], X_lda[:, 1], c=y, cmap='viridis', alpha=0.6)
        plt.xlabel('LDA Component 1')
        plt.ylabel('LDA Component 2')
        plt.legend(*scatter.legend_elements(), title='Classes')

    plt.title('LDA Projection')
    plt.show()

    return X_lda
