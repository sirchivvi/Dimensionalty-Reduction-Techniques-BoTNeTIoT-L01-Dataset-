def apply_tsne_optimized(X, y, sample_size=5000, perplexity=15, n_iter=250):
    """
    Optimized t-SNE implementation with sensible defaults

    Parameters:
    X : features array
    y : target labels
    sample_size : number of samples to use (default: 5000)
    perplexity : t-SNE parameter (default: 15)
    n_iter : iterations (default: 250)
    """
    # Step 1: Subsample if dataset is large
    if len(X) > sample_size:
        X_sample, _, y_sample, _ = train_test_split(
            X, y,
            train_size=sample_size,
            stratify=y,
            random_state=42
        )
        print(f"Using subsample of {sample_size} points for t-SNE")
    else:
        X_sample, y_sample = X, y

    # Step 2: Set optimized parameters
    tsne = TSNE(
        n_components=2,
        perplexity=perplexity,
        n_iter=n_iter,
        random_state=42,
        verbose=1  # Show progress
    )

    # Step 3: Fit and transform
    print("Running t-SNE...")
    start_time = time.time()
    X_tsne = tsne.fit_transform(X_sample)
    print(f"t-SNE completed in {(time.time()-start_time)/60:.1f} minutes")

    # Visualization
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=y_sample, cmap='viridis', alpha=0.6)
    plt.xlabel('t-SNE 1')
    plt.ylabel('t-SNE 2')
    plt.title(f't-SNE (perplexity={perplexity}, n_iter={n_iter})')
    plt.legend(*scatter.legend_elements(), title='Classes')
    plt.show()

    return X_tsne
