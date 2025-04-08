import numpy as np
import matplotlib.pyplot as plt
import random
from typing import Tuple, Union
from PIL import Image


def generate_maze(grid_size: Union[int, Tuple[int, int]]):
    if isinstance(grid_size, int):
        rows, cols = grid_size, grid_size
    else:
        rows, cols = grid_size

    # Create an empty grid: 0 = wall, 1 = path
    maze = np.zeros((rows, cols), dtype=int)

    # Start and end positions
    start = (random.randint(0, rows - 1), random.randint(0, cols - 1))
    end = (random.randint(0, rows - 1), random.randint(0, cols - 1))
    while end == start or max(abs(start[0] - end[0]), abs(start[1] - end[1])) < 2:
        end = (random.randint(0, rows - 1), random.randint(0, cols - 1))

    # Carve a path using DFS from start to end
    path = []
    visited = set()

    def cnt_path_near(cell):
        cnt = 0
        for dr, dc in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
            nr, nc = cell[0] + dr, cell[1] + dc
            if 0 <= nr < rows and 0 <= nc < cols and (nr, nc) in path:
                cnt += 1
        return cnt

    def dfs(cell):
        if cell == end:
            path.append(cell)
            return True
        visited.add(cell)
        path.append(cell)
        directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
        random.shuffle(directions)
        for dr, dc in directions:
            nr, nc = cell[0] + dr, cell[1] + dc
            next_cell = (nr, nc)
            if 0 <= nr < rows and 0 <= nc < cols and next_cell not in visited:
                if cnt_path_near(next_cell) > 1:
                    continue
                if dfs(next_cell):
                    return True
        path.pop()
        return False

    dfs(start)

    while cnt_path_near(start) > 1:
        start = path.pop(0)

    while cnt_path_near(end) > 1:
        end = path.pop()

    # Mark the solution path in the maze
    for r, c in path:
        maze[r, c] = 1

    maze = 1 - maze  # Invert the maze: 1 = wall, 0 = path
    maze[start] = 2  # Start point
    maze[end] = 3  # End point

    return maze, path, start, end


def draw_maze(maze, path, start, end, filename="maze.png"):
    rows, cols = maze.shape
    img = np.zeros((rows, cols, 3), dtype=np.uint8)

    for r in range(rows):
        for c in range(cols):
            if maze[r, c] == 0:
                img[r, c] = [255, 255, 255]  # White path

    for r, c in path:
        img[r, c] = [0, 0, 255]  # Blue path

    sr, sc = start
    er, ec = end
    img[sr, sc] = [0, 255, 0]  # Green start
    img[er, ec] = [255, 0, 0]  # Red goal

    plt.figure(figsize=(5, 5))
    plt.imshow(img, interpolation="nearest")
    plt.axis("off")
    plt.savefig(filename, bbox_inches="tight", pad_inches=0)
    plt.close()


# Example usage
maze, path, start, end = generate_maze((5, 5))
draw_maze(maze, path, start, end)
