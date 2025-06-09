Dimensionality Reduction for IoT Intrusion Detection
Overview

This project applies and compares various dimensionality reduction techniques—PCA, LDA, Truncated SVD, t-SNE, and MDS—on a high-dimensional cybersecurity dataset to improve classification performance and support data visualization.

The dataset used is the Bot-IoT NeTIoT-L01 labeled dataset, consisting of over 2.4 million records capturing simulated IoT network traffic, including attack and benign flows.
Objectives

    Reduce the dimensionality of high-volume IoT network data.

    Evaluate classification accuracy after reduction.

    Visualize cluster separability using 2D projections.

    Analyze computational trade-offs across techniques.

Dataset

Source: BotNeTIoT-L01_label_NoDuplicates.csv
Records: ~2.4 million
Features: 25+ (entropy, packet flow stats, mutual information, etc.)
Label Classes: Multiple IoT attack types (DDoS, DoS, reconnaissance) and normal traffic
Techniques Compared
Technique	Type	Purpose	Accuracy (RF Classifier)	Notes
PCA	Unsupervised	Maximize variance	99.96%	Best performing
LDA	Supervised	Maximize class separability	99.66%	Affected by class imbalance
SVD	Unsupervised	Matrix decomposition	99.96%	Matched PCA
t-SNE	Unsupervised	Visualization only (nonlinear)	Visual only	Highlights patterns, not good for modeling
MDS	Unsupervised	Preserve pairwise distances	Skipped	Computationally infeasible
Methodology
1. Preprocessing

    Dropped missing values

    Label encoded target variable

    Scaled features using StandardScaler

2. Dimensionality Reduction

Implemented the following using sklearn:

    PCA

    LDA

    TruncatedSVD

    TSNE (optimized for large datasets)

    MDS (excluded due to 23+ TB memory requirement)

3. Evaluation

    Classification using Random Forest

    train_test_split with 70/30 split

    Evaluation using accuracy score and classification_report

4. Visualization

    2D scatter plots with color-coded classes

    Explained variance plots (PCA, SVD)

    Cluster inspection (t-SNE)

Results & Insights

    PCA & SVD performed the best (99.96% accuracy), making them ideal for both dimensionality reduction and classification in high-volume intrusion datasets.

    LDA worked well with labels but was less robust in presence of class imbalance.

    t-SNE proved invaluable for visual analytics, despite poor classification alignment.

    MDS was computationally impractical (theoretical 23 TB memory and 80+ days processing).

Visual Examples

    Add .png plots if available:

/visualizations/
  pca_projection.png
  lda_projection.png
  tsne_clusters.png
  explained_variance_pca.png

Sample Usage (Colab Compatible)

from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Preprocess
X_scaled = StandardScaler().fit_transform(X)

# PCA
X_pca = PCA(n_components=2).fit_transform(X_scaled)

# Train model
X_train, X_test, y_train, y_test = train_test_split(X_pca, y, test_size=0.3)
model = RandomForestClassifier().fit(X_train, y_train)

print("Accuracy:", model.score(X_test, y_test))

Technologies Used

    Python 3.10

    Scikit-learn

    Pandas, NumPy

    Seaborn, Matplotlib

    Jupyter/Colab
