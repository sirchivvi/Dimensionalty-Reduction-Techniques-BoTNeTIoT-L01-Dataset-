def main():
    try:
        # Load and preprocess data
        filepath = '/content/BotNeTIoT-L01_label_NoDuplicates.csv'
        print("Loading data...")
        df = load_data(filepath)
        X, y = preprocess_data(df)

        # Perform EDA
        print("\nPerforming EDA...")
        perform_eda(df)

        # Initialize techniques dictionary
        techniques = {}

        # Apply dimensionality reduction techniques with error handling
        print("\nApplying dimensionality reduction techniques...")

        # 1. PCA
        print("\n1. Running PCA...")
        techniques['PCA'] = apply_pca(X, y)

        # 2. LDA (only if we have multiple classes)
        if len(np.unique(y)) > 1:
            print("\n2. Running LDA...")
            techniques['LDA'] = apply_lda(X, y)
        else:
            print("\n2. Skipping LDA - Only one class present")

        # 3. SVD
        print("\n3. Running SVD...")
        techniques['SVD'] = apply_svd(X, y)

        # 4. MDS (only for small datasets)
        if len(X) <= 10000:
            print("\n4. Running MDS...")
            techniques['MDS'] = apply_mds(X, y)
        else:
            print("\n4. Skipping MDS - Dataset too large (n > 5000)")

        # 5. t-SNE (optimized implementation)
        print("\n5. Running t-SNE with optimized parameters...")
        techniques['t-SNE'] = apply_tsne_optimized(X, y)

        # Evaluate models
        print("\nEvaluating models...")
        accuracies = {}
        for name, X_transformed in techniques.items():
            if X_transformed is not None:  # Skip if technique failed
                try:
                    acc = evaluate_model(X_transformed, y, name)
                    accuracies[name] = acc
                except Exception as e:
                    print(f"Failed to evaluate {name}: {str(e)}")

        # Compare techniques
        print("\nComparing techniques...")
        compare_techniques(accuracies)

    except Exception as e:
        print(f"Main execution failed: {str(e)}")

if __name__ == '__main__':
    import numpy as np
    import time
    from sklearn.manifold import TSNE, MDS
    from sklearn.decomposition import PCA, TruncatedSVD
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
    from sklearn.model_selection import train_test_split
    main()
