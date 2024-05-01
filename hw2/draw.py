import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns


def load_and_process_data(filepath):
    # Load the CSV file into a DataFrame
    data = pd.read_csv(filepath)
    # Convert time columns to float if they are not already
    data['DSU Time'] = data['DSU Time'].astype(float)
    data['Recalculation Time'] = data['Recalculation Time'].astype(float)
    return data


def plot_3d(data, z_axis='DSU Time'):
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Create 3D scatter plot
    scatter = ax.scatter(data['Operations'], data['Nodes'], data[z_axis], c=data[z_axis], cmap='viridis')

    # Add labels and title
    ax.set_xlabel('Number of Operations')
    ax.set_ylabel('Number of Nodes')
    ax.set_zlabel(z_axis)
    ax.set_title(f'3D Plot of {z_axis} vs. Number of Operations and Nodes')

    # Color bar
    color_bar = fig.colorbar(scatter, ax=ax, shrink=0.5, aspect=10)
    color_bar.set_label(z_axis)

    # plt.show()
    plt.savefig(f'3D Plot of {z_axis} vs Number of Operations and Nodes.png')


def plot_performance(data):
    # DSU Time vs. Number of Nodes
    plt.figure(figsize=(12, 6))
    sns.lineplot(x='Nodes', y='DSU Time', hue='Scenario', data=data)
    plt.title('DSU Time vs. Number of Nodes')
    plt.xlabel('Number of Nodes')
    plt.ylabel('DSU Time (seconds)')
    plt.legend(title='Scenario')
    plt.grid(True)
    # plt.show()
    plt.savefig("DSU Time vs Number of Nodes.png")

    # Recalculation Time vs. Number of Nodes
    plt.figure(figsize=(12, 6))
    sns.lineplot(x='Nodes', y='Recalculation Time', hue='Scenario', data=data)
    plt.title('Recalculation Time vs. Number of Nodes')
    plt.xlabel('Number of Nodes')
    plt.ylabel('Recalculation Time (seconds)')
    plt.legend(title='Scenario')
    plt.grid(True)
    # plt.show()
    plt.savefig('Recalculation Time vs Number of Nodes.png')

    # DSU Time vs. Number of Operations
    plt.figure(figsize=(12, 6))
    sns.lineplot(x='Operations', y='DSU Time', hue='Scenario', data=data)
    plt.title('DSU Time vs. Number of Operations')
    plt.xlabel('Number of Operations')
    plt.ylabel('DSU Time (seconds)')
    plt.legend(title='Scenario')
    plt.grid(True)
    # plt.show()
    plt.savefig('DSU Time vs Number of Operations.png')

    # Recalculation Time vs. Number of Operations
    plt.figure(figsize=(12, 6))
    sns.lineplot(x='Operations', y='Recalculation Time', hue='Scenario', data=data)
    plt.title('Recalculation Time vs. Number of Operations')
    plt.xlabel('Number of Operations')
    plt.ylabel('Recalculation Time (seconds)')
    plt.legend(title='Scenario')
    plt.grid(True)
    # plt.show()
    plt.savefig('Recalculation Time vs Number of Operations.png')

    # Correlation matrix
    plt.figure(figsize=(8, 6))
    correlation_matrix = data[['Nodes', 'Operations', 'DSU Time', 'Recalculation Time']].corr()
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
    plt.title('Correlation Matrix')
    # plt.show()
    plt.savefig('Correlation Matrix.png')

    plot_3d(data, 'DSU Time')  # 3D plot for DSU Time
    plot_3d(data, 'Recalculation Time')


def main():
    # Update the filepath below with your actual file location
    filepath = 'dsu_performance.csv'
    data = load_and_process_data(filepath)
    plot_performance(data)


if __name__ == "__main__":
    main()
