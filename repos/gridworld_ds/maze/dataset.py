import os
import os.path as osp
import numpy as np
import tqdm
import json

from generate import generate_maze, draw_maze


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


def generate_dataset(grid_size=10, num_samples=900, output_dir="data"):
    """
    Generate a dataset of mazes and save them to a JSON file.
    Args:
        num_samples (int): Number of mazes to generate.
        grid_size (tuple): Size of the maze grid (rows, cols).
        output_dir (str): Directory to save the dataset.
    """
    if not isinstance(grid_size, tuple):
        grid_size = (grid_size, grid_size)
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(osp.join(output_dir, "images"), exist_ok=True)
    os.makedirs(osp.join(output_dir, "images", f"{grid_size[0]}x{grid_size[1]}"), exist_ok=True)

    with open(osp.join(output_dir, f"{grid_size[0]}x{grid_size[1]}.jsonl"), "a") as f:
        for i in tqdm.tqdm(range(num_samples)):
            maze, path, start, end = generate_maze(grid_size)
            image_filename = osp.join(output_dir, "images", f"{grid_size[0]}x{grid_size[1]}", f"{i:04d}.png")
            draw_maze(maze, path, start, end, image_filename)

            category = ["right", "total", "displc"][i // 300]

            if category == "right":
                right_turns = count_right_turns(path)
                answer = right_turns
                template_path = "/mnt/ssd/Projects/lmms-gridworld/repos/gridworld_ds/maze/templates/turncnt_right.txt"
                choices = make_options(answer)
                choices = [str(c) for c in choices]
                qid = "turncnt_right"

            elif category == "total":
                total_turns = count_left_turns(path) + count_right_turns(path)
                answer = total_turns
                template_path = "/mnt/ssd/Projects/lmms-gridworld/repos/gridworld_ds/maze/templates/turncnt_total.txt"
                choices = make_options(answer)
                choices = [str(c) for c in choices]
                qid = "turncnt_total"
            else:
                answer = "Yes" if start[0] == end[0] else "No"
                template_path = "/mnt/ssd/Projects/lmms-gridworld/repos/gridworld_ds/maze/templates/displc_horiz.txt"
                choices = ["Yes", "No"]
                qid = "displc_horiz"

            with open(template_path, "r") as template_file:
                template = template_file.read()

            f.write(
                json.dumps(
                    {
                        "question_id": f"{qid}_{grid_size[0]}x{grid_size[1]}_{(i):04d}",
                        "image": image_filename,
                        "question": template,
                        "choices": choices,
                        "answer": str(answer),
                        "grid_size": grid_size,
                        "metadata": {
                            "map": maze.tolist(),
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
    generate_dataset(grid_size=(5, 5), num_samples=900, output_dir="data")
