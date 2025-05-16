# This script performs data analysis and visualization on the Iris dataset.
# It includes error handling, data exploration, basic analysis, and various plots.
# 1. Import Libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris

sns.set(style='whitegrid')

# 2. Load Dataset with Error Handling
try:
    iris = load_iris()
    df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
    df['species'] = pd.Categorical.from_codes(iris.target, iris.target_names)

    print("Dataset loaded successfully!")
    print(df.head())

    # 3. Data Exploration
    print(df.info())
    print(df.isnull().sum())

    # 4. Basic Data Analysis
    print(df.describe())
    print(df.groupby('species').mean())

    # 5. Data Visualization

    # Line Chart
    plt.figure(figsize=(10, 5))
    plt.plot(df.index, df['petal length (cm)'],
             label='Petal Length', color='green')
    plt.title('Simulated Petal Length Trend Over Index')
    plt.xlabel('Index')
    plt.ylabel('Petal Length (cm)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("plot1_line_chart.png")
    plt.close()

    # Bar Chart
    plt.figure(figsize=(8, 5))
    sns.barplot(x='species', y='petal length (cm)', data=df, palette='Set2')
    plt.title('Average Petal Length per Species')
    plt.xlabel('Species')
    plt.ylabel('Petal Length (cm)')
    plt.tight_layout()
    plt.savefig("plot2_bar_chart.png")
    plt.close()

    # Histogram
    plt.figure(figsize=(8, 5))
    sns.histplot(df['sepal width (cm)'], kde=True, bins=15, color='skyblue')
    plt.title('Distribution of Sepal Width')
    plt.xlabel('Sepal Width (cm)')
    plt.ylabel('Frequency')
    plt.tight_layout()
    plt.savefig("plot3_histogram.png")
    plt.close()

    # Scatter Plot
    plt.figure(figsize=(8, 5))
    sns.scatterplot(x='sepal length (cm)', y='petal length (cm)',
                    hue='species', data=df, palette='Dark2')
    plt.title('Sepal Length vs Petal Length by Species')
    plt.xlabel('Sepal Length (cm)')
    plt.ylabel('Petal Length (cm)')
    plt.legend(title='Species')
    plt.tight_layout()
    plt.savefig("plot4_scatter_plot.png")
    plt.close()

except Exception as e:
    print(f"An error occurred: {e}")
