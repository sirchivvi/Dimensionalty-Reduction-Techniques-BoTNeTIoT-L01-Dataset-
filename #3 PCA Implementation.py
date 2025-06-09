def apply_pca(X, y, n_components=2):
    """
    Apply Principal Component Analysis (PCA) to the dataset.

    Parameters:
    X (array): Scaled features
    y (array): Labels
    n_components (int): Number of components to keep

    Returns:
    array: Transformed features
    """
    print("\n=== PCA ===")
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X)

    # Plot explained variance ratio
    plt.figure(figsize=(8, 6))
    plt.bar(range(pca.n_components_), pca.explained_variance_ratio_)
    plt.xlabel('Principal Component')
    plt.ylabel('Variance Explained')
    plt.title('PCA Explained Variance Ratio')
    plt.show()

    # Plot PCA components
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='viridis', alpha=0.6)
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.title('PCA Projection')
    plt.legend(*scatter.legend_elements(), title='Classes')
    plt.show()

    return X_pca
