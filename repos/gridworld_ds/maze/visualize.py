import numpy as np
from matplotlib.colors import ListedColormap
import os

import matplotlib.pyplot as plt


def create_maze_visualization(maze, colormap=None, figsize=(10, 10)):
    """
    Create a figure with a 2D grid array visualized as an image with different colored tiles.

    Parameters:
    -----------
    maze : 2D numpy array or list of lists
        The maze/grid to visualize. Different values represent different cell types.
    colormap : dict or None, optional
        Dictionary mapping cell values to colors. If None, a default colormap will be used.
    figsize : tuple, optional
        Size of the figure (width, height) in inches.

    Returns:
    --------
    fig : matplotlib.figure.Figure
        The figure object containing the maze visualization
    """
    # Convert to numpy array if not already
    if not isinstance(maze, np.ndarray):
        maze = np.array(maze)

    # Default colormap if none provided
    if colormap is None:
        # Default colors for common maze elements:
        # 0: path (white), 1: wall (black), 2: start (green), 3: goal (red)
        colormap = {
            0: "white",  # Path
            1: "black",  # Wall
            2: "green",  # Start
            3: "red",  # Goal
        }

    # Get unique values in the maze
    unique_values = np.unique(maze)
    colors = [colormap.get(val, "gray") for val in unique_values]
    cmap = ListedColormap(colors)

    # Create figure and plot maze
    fig = plt.figure(figsize=figsize)
    plt.imshow(maze, cmap=cmap, interpolation="nearest")

    # Remove axis ticks
    plt.xticks([])
    plt.yticks([])

    # Add grid lines
    plt.grid(True, color="gray", linestyle="-", linewidth=0.5, alpha=0.3)

    return fig


def save_visualization(fig, output_path="maze_visualization.png"):
    """
    Save a matplotlib figure to a file.

    Parameters:
    -----------
    fig : matplotlib.figure.Figure
        The figure to save
    output_path : str, optional
        Path where the image will be saved.

    Returns:
    --------
    str
        The output path where the image was saved
    """
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    # Save the visualization
    fig.savefig(output_path, bbox_inches="tight", dpi=100)
    plt.close(fig)

    print(f"Maze visualization saved to {output_path}")
    return output_path


def visualize_maze(maze, output_path="maze_visualization.png", colormap=None, figsize=(10, 10)):
    """
    Visualize a 2D grid array as an image with different colored tiles and save it.

    Parameters:
    -----------
    maze : 2D numpy array or list of lists
        The maze/grid to visualize. Different values represent different cell types.
    output_path : str, optional
        Path where the image will be saved.
    colormap : dict or None, optional
        Dictionary mapping cell values to colors. If None, a default colormap will be used.
    figsize : tuple, optional
        Size of the figure (width, height) in inches.
    """
    fig = create_maze_visualization(maze, colormap, figsize)
    return save_visualization(fig, output_path)


# Example usage
if __name__ == "__main__":
    # Example maze: 0=path, 1=wall, 2=start, 3=goal
    example_maze = [
        [1, 1, 1, 1, 1, 1, 1],
        [1, 2, 0, 0, 0, 0, 1],
        [1, 1, 1, 0, 1, 0, 1],
        [1, 0, 0, 0, 1, 0, 1],
        [1, 0, 1, 1, 1, 0, 1],
        [1, 0, 0, 0, 0, 3, 1],
        [1, 1, 1, 1, 1, 1, 1],
    ]

    # Visualize with default settings
    visualize_maze(example_maze)

    # You can also customize the colors
    custom_colors = {
        0: "lightblue",  # Path
        1: "darkblue",  # Wall
        2: "lime",  # Start
        3: "red",  # Goal
    }
    visualize_maze(example_maze, "custom_maze.png", custom_colors)
