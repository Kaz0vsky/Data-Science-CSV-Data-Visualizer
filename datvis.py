import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import os
import numpy as np
from pandas.plotting import andrews_curves, parallel_coordinates, radviz
from sklearn.decomposition import PCA

def plot_histograms(df, output_dir):
    """Generates histograms for each numerical column."""
    for column in df.select_dtypes(include=[float, int]).columns:
        plt.figure(figsize=(8, 6))
        sns.histplot(df[column], kde=True, color='skyblue')
        plt.title(f'Histogram of {column}')
        plt.xlabel(column)
        plt.ylabel('Frequency')
        plt.savefig(os.path.join(output_dir, f'{column}_histogram.png'))
        plt.close()

def plot_scatter_matrix(df, output_dir):
    """Generates a scatter plot matrix for numerical columns."""
    sns.pairplot(df.select_dtypes(include=[float, int]), diag_kind='kde', plot_kws={'alpha': 0.6})
    plt.savefig(os.path.join(output_dir, 'scatter_matrix.png'))
    plt.close()

def plot_correlation_heatmap(df, output_dir):
    """Generates a heatmap to visualize correlations between numerical columns."""
    # Select only numerical columns
    numeric_df = df.select_dtypes(include=[float, int])
    
    if numeric_df.empty:
        print("No numerical data available for correlation heatmap.")
        return
    
    plt.figure(figsize=(10, 8))
    correlation_matrix = numeric_df.corr()
    mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
    sns.heatmap(correlation_matrix, mask=mask, annot=True, cmap='coolwarm', fmt=".2f")
    plt.title('Correlation Heatmap')
    plt.savefig(os.path.join(output_dir, 'correlation_heatmap.png'))
    plt.close()

def plot_categorical_counts(df, output_dir):
    """Generates bar plots for categorical columns."""
    for column in df.select_dtypes(include=['object']).columns:
        plt.figure(figsize=(8, 6))
        sns.countplot(y=column, data=df, palette="viridis", order=df[column].value_counts().index, hue=column, dodge=False, legend=False)
        plt.title(f'Count Plot of {column}')
        plt.xlabel('Count')
        plt.ylabel(column)
        plt.savefig(os.path.join(output_dir, f'{column}_count_plot.png'))
        plt.close()

def plot_andrews_curves(df, output_dir):
    """Generates Andrews Curves plot for visualizing high-dimensional data."""
    plt.figure(figsize=(10, 6))
    andrews_curves(df, class_column=df.select_dtypes(include=['object']).columns[0], colormap='cool')
    plt.title('Andrews Curves')
    plt.savefig(os.path.join(output_dir, 'andrews_curves.png'))
    plt.close()

def plot_parallel_coordinates(df, output_dir):
    """Generates Parallel Coordinates plot to visualize multi-dimensional data."""
    plt.figure(figsize=(10, 6))
    parallel_coordinates(df, class_column=df.select_dtypes(include=['object']).columns[0], colormap='winter')
    plt.title('Parallel Coordinates')
    plt.savefig(os.path.join(output_dir, 'parallel_coordinates.png'))
    plt.close()

def plot_radviz(df, output_dir):
    """Generates RadViz plot to visualize multivariate data."""
    plt.figure(figsize=(8, 6))
    radviz(df, class_column=df.select_dtypes(include=['object']).columns[0], colormap='plasma')
    plt.title('RadViz Plot')
    plt.savefig(os.path.join(output_dir, 'radviz_plot.png'))
    plt.close()

def plot_pca_variance(df, output_dir):
    """Performs PCA and plots explained variance for each component."""
    numeric_df = df.select_dtypes(include=[float, int]).dropna()
    pca = PCA()
    pca.fit(numeric_df)
    plt.figure(figsize=(10, 6))
    plt.plot(np.cumsum(pca.explained_variance_ratio_), marker='o', linestyle='--', color='b')
    plt.xlabel('Number of Components')
    plt.ylabel('Cumulative Explained Variance')
    plt.title('PCA Explained Variance')
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, 'pca_variance.png'))
    plt.close()

def generate_visualizations(file_path):
    # Create output directory
    output_dir = "advanced_visualization_output"
    os.makedirs(output_dir, exist_ok=True)

    # Load data
    df = pd.read_csv(file_path)

    # Generate visualizations
    plot_histograms(df, output_dir)
    plot_scatter_matrix(df, output_dir)
    plot_correlation_heatmap(df, output_dir)
    plot_categorical_counts(df, output_dir)
    plot_andrews_curves(df, output_dir)
    plot_parallel_coordinates(df, output_dir)
    plot_radviz(df, output_dir)
    plot_pca_variance(df, output_dir)

    print(f"Advanced visualizations saved in {output_dir} directory.")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python3 datvis.py <path_to_csv_file>")
    else:
        file_path = sys.argv[1]
        generate_visualizations(file_path)
