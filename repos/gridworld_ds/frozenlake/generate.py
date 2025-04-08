import os
import json
import tqdm
import os.path as osp
import numpy as np
import gymnasium as gym
from gymnasium.envs.toy_text.frozen_lake import generate_random_map
import numpy as np
from matplotlib import animation
from matplotlib import pyplot as plt


def find_shortest_paths(desc):
    """
    Find all shortest paths from 'S' to 'G' using BFS.

    Args:
        desc: Description of the FrozenLake map

    Returns:
        A list of paths, where each path is a list of (row, col) coordinates
    """
    if not desc:
        return []

    # Convert desc to a 2D array for easier access
    grid = [list(row) for row in desc]
    rows, cols = len(grid), len(grid[0])

    # Find start and goal positions
    start, goal = None, None
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] == "S":
                start = (r, c)
            elif grid[r][c] == "G":
                goal = (r, c)

    if not start or not goal:
        return []

    # Directions: up, right, down, left
    directions = [(-1, 0), (0, 1), (1, 0), (0, -1)]

    # BFS to find shortest paths
    level = 0
    current_level = [start]  # Nodes at current level
    paths = {start: [[start]]}  # To track unique paths
    levels = {start: 0}  # To track levels of nodes

    while current_level:
        next_level = []

        for r, c in current_level:
            if (r, c) == goal:
                break

            for dr, dc in directions:
                nr, nc = r + dr, c + dc

                if 0 <= nr < rows and 0 <= nc < cols and grid[nr][nc] != "H":
                    new_paths = [p + [(nr, nc)] for p in paths[(r, c)]]
                    if (nr, nc) not in levels:
                        levels[(nr, nc)] = level + 1
                        paths[(nr, nc)] = new_paths
                        next_level.append((nr, nc))
                    elif levels[(nr, nc)] == level + 1:
                        paths[(nr, nc)].extend(new_paths)

        current_level = next_level
        level += 1

    return paths[goal]


def visualize_path_on_image(grid_map, path, vis_image):
    """
    Visualize a path on top of a grid image.

    Args:
        grid_map: 2D array representing the grid
        path: Sequence of (row, col) coordinates representing the path
        vis_image: Image array of the grid visualization

    Returns:
        Image array with the path visualized on top of vis_image
    """
    # Create a copy of the visualization image to avoid modifying the original
    result_image = np.copy(vis_image)

    if not path:
        return result_image

    # Determine the size of each cell in the visualization
    h, w, _ = result_image.shape
    rows, cols = len(grid_map), len(grid_map[0])
    cell_h, cell_w = h // rows, w // cols

    # Draw path with red lines
    for i in range(len(path) - 1):
        r1, c1 = path[i]
        r2, c2 = path[i + 1]

        # Calculate center points of the cells
        y1, x1 = int((r1 + 0.5) * cell_h), int((c1 + 0.5) * cell_w)
        y2, x2 = int((r2 + 0.5) * cell_h), int((c2 + 0.5) * cell_w)

        # Draw line between centers
        rr, cc = draw_line(y1, x1, y2, x2)

        # Filter points that are within image boundaries
        valid_indices = (0 <= rr) & (rr < h) & (0 <= cc) & (cc < w)
        rr, cc = rr[valid_indices], cc[valid_indices]

        # Draw red line (RGB format)
        result_image[rr, cc, 0] = 255  # Red channel
        result_image[rr, cc, 1] = 0  # Green channel
        result_image[rr, cc, 2] = 0  # Blue channel

    return result_image


def draw_line(y0, x0, y1, x1):
    """
    Draw a line between two points using Bresenham's algorithm.
    Returns arrays of y, x coordinates.
    """
    steep = abs(y1 - y0) > abs(x1 - x0)
    if steep:
        x0, y0 = y0, x0
        x1, y1 = y1, x1

    if x0 > x1:
        x0, x1 = x1, x0
        y0, y1 = y1, y0

    dx = x1 - x0
    dy = abs(y1 - y0)
    error = dx // 2
    y = y0

    if y0 < y1:
        ystep = 1
    else:
        ystep = -1

    points_x = []
    points_y = []

    for x in range(x0, x1 + 1):
        if steep:
            points_y.append(x)
            points_x.append(y)
        else:
            points_y.append(y)
            points_x.append(x)

        error -= dy
        if error < 0:
            y += ystep
            error += dx

    return np.array(points_y), np.array(points_x)


def count_right_turns(path):
    """
    Count the number of right turns in a path.

    Args:
        path: A list of (row, col) coordinates representing a path

    Returns:
        Number of right turns in the path
    """
    if len(path) < 3:
        return 0

    right_turns = 0

    # Define direction vectors: up, right, down, left
    directions = [(-1, 0), (0, 1), (1, 0), (0, -1)]

    for i in range(len(path) - 2):
        # Calculate direction vectors between consecutive points
        d1 = (path[i + 1][0] - path[i][0], path[i + 1][1] - path[i][1])
        d2 = (path[i + 2][0] - path[i + 1][0], path[i + 2][1] - path[i + 1][1])

        # Find the indices of these directions in our directions list
        try:
            d1_idx = directions.index(d1)
            d2_idx = directions.index(d2)

            # A right turn means the next direction is 90 degrees clockwise
            # In our directions list, that means the index increases by 1 (with wraparound)
            if (d1_idx + 1) % 4 == d2_idx:
                right_turns += 1
        except ValueError:
            # If a direction isn't one of our cardinal directions, skip it
            continue

    return right_turns


def count_left_turns(path):
    """
    Count the number of left turns in a path.

    Args:
        path: A list of (row, col) coordinates representing a path

    Returns:
        Number of left turns in the path
    """
    if len(path) < 3:
        return 0

    left_turns = 0

    # Define direction vectors: up, right, down, left
    directions = [(-1, 0), (0, 1), (1, 0), (0, -1)]

    for i in range(len(path) - 2):
        # Calculate direction vectors between consecutive points
        d1 = (path[i + 1][0] - path[i][0], path[i + 1][1] - path[i][1])
        d2 = (path[i + 2][0] - path[i + 1][0], path[i + 2][1] - path[i + 1][1])

        # Find the indices of these directions in our directions list
        try:
            d1_idx = directions.index(d1)
            d2_idx = directions.index(d2)

            # A left turn means the next direction is 90 degrees counterclockwise
            # In our directions list, that means the index decreases by 1 (with wraparound)
            if (d1_idx - 1) % 4 == d2_idx:
                left_turns += 1
        except ValueError:
            # If a direction isn't one of our cardinal directions, skip it
            continue

    return left_turns


def make_options(answer):
    """
    Generate multiple-choice options for the question.

    Args:
        answer: The correct answer

    Returns:
        A list of options including the correct answer and some random choices
    """
    options = [answer]
    while len(options) < 4:
        option = np.random.randint(0, 10)
        if option not in options:
            options.append(option)
    np.random.shuffle(options)
    return options


def generate_dataset(grid_size, num_samples, output_path="data"):
    """
    Generate a dataset of FrozenLake maps and their shortest paths.

    Args:
        grid_size: Size of the grid (e.g., 4 for 4x4)
        num_samples: Number of samples to generate
    """
    # Open jsonl file for writing
    os.makedirs(output_path, exist_ok=True)
    os.makedirs(osp.join(output_path, "images"), exist_ok=True)
    os.makedirs(osp.join(output_path, "images", f"{grid_size}x{grid_size}"), exist_ok=True)
    with open(osp.join(output_path, f"{grid_size}x{grid_size}.jsonl"), "w") as f:
        for i in tqdm.trange(num_samples):
            desc = generate_random_map(grid_size)
            env = gym.make("FrozenLake-v1", render_mode="rgb_array", desc=desc, is_slippery=False)
            env.reset()

            paths = find_shortest_paths(desc)

            frame = env.render()
            path = paths[np.random.randint(len(paths))]
            path_frame_image = visualize_path_on_image(desc, path, frame)

            image_subdir = osp.join(output_path, "images", f"{grid_size}x{grid_size}")

            image_path = osp.join(image_subdir, f"{i:04d}.png")
            plt.imsave(image_path, path_frame_image)

            # category = np.random.choice(["left", "right", "total"])
            category = ["left", "right", "total"][i // 300]

            if category == "left":
                left_turns = count_left_turns(path)
                answer = left_turns
            elif category == "right":
                right_turns = count_right_turns(path)
                answer = right_turns
            else:
                total_turns = count_left_turns(path) + count_right_turns(path)
                answer = total_turns

            choices = make_options(answer)
            choices = [str(c) for c in choices]

            f.write(
                json.dumps(
                    {
                        "question_id": f"turncnt_{category}_{grid_size}x{grid_size}_{(i):04d}",
                        "image": image_path,
                        "question": f"The image shows the elf's path in a FrozenLake grid. The elf must reach the goal (gift) by moving only to adjacent tiles that are not holes, walls, or out of bounds. Movements are allowed in four directions: up, down, left, and right. The path is marked in red.\nNow answer the following question: How many {category} turns are there in the path?",
                        "choices": choices,
                        "answer": answer,
                        "grid_size": grid_size,
                        "metadata": {
                            "map": desc,
                            "path": path,
                            "turns": {
                                "left": count_left_turns(path),
                                "right": count_right_turns(path),
                                "total": count_left_turns(path) + count_right_turns(path),
                            },
                        },
                    }
                )
                + "\n"
            )


if __name__ == "__main__":
    generate_dataset(grid_size=3, num_samples=900)
