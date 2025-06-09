def apply_mds(X, y, n_components=2):
    """
    Apply Multidimensional Scaling (MDS) to the dataset.

    Parameters:
    X (array): Scaled features
    y (array): Labels
    n_components (int): Number of components to keep

    Returns:
    array: Transformed features
    """
    print("\n=== MDS ===")
    mds = MDS(n_components=n_components, random_state=42)
    X_mds = mds.fit_transform(X)

    # Plot MDS components
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(X_mds[:, 0], X_mds[:, 1], c=y, cmap='viridis', alpha=0.6)
    plt.xlabel('MDS Component 1')
    plt.ylabel('MDS Component 2')
    plt.title('MDS Projection')
    plt.legend(*scatter.legend_elements(), title='Classes')
    plt.show()

    return X_mds
