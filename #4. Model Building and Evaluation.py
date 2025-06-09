def evaluate_model(X, y, technique_name):
    """
    Evaluate a Random Forest classifier on the transformed data.

    Parameters:
    X (array): Transformed features
    y (array): Labels
    technique_name (str): Name of the dimensionality reduction technique

    Returns:
    float: Accuracy score
    """
    print(f"\nEvaluating model with {technique_name} features...")

    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    # Create and train model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Make predictions and evaluate
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\nClassification Report for {technique_name}:")
    print(classification_report(y_test, y_pred))
    print(f"Accuracy: {accuracy:.4f}")

    return accuracy
