from concurrent.futures import ProcessPoolExecutor
import numpy as np
from copy import deepcopy
from time import sleep, time
import matplotlib.pyplot as plt
from IPython.display import clear_output, display


def turn(direction: np.ndarray) -> np.ndarray:
    return np.array([direction[1], -direction[0]])


def is_outside(board, c_pos, c_dir) -> bool:
    return any([
        (c_dir[0] == -1 and c_pos[0] == 0),
        (c_dir[1] == -1 and c_pos[1] == 0),
        (c_dir[0] == 1 and c_pos[0] == board.shape[0]-1),
        (c_dir[1] == 1 and c_pos[1] == board.shape[1]-1)
    ])


def compute_route(arr, c_pos, perform_plot: bool):
    # get starting direction
    c_dir = np.array([-1, 0])

    # convert starting pos to 1
    arr[*c_pos] = 1

    # setup plot figure
    if perform_plot:
        fig, ax = plt.subplots(figsize=(8, 8))
        mat = ax.matshow(arr, cmap=plt.cm.Blues)
        plt.axis('off')

    is_looping = False
    visited_obstacles = []
    while not is_outside(arr, c_pos, c_dir):
        next_pos = c_pos + c_dir
        if arr[*next_pos] == 3:
            obsactle = [*list(next_pos), *list(c_dir)]
            if obsactle in visited_obstacles:
                is_looping = True
                break
            visited_obstacles.append(obsactle)

            # update direction and next position
            c_dir = turn(c_dir)
            next_pos = c_pos
            continue

        # update current position to be next position
        c_pos = next_pos

        # update current position to be X
        arr[*c_pos] = 1 

        # update plot
        if perform_plot:
            mat.set_data(arr)
            clear_output(wait=True)
            display(fig)
            # sleep(0.01)
    plt.close()
    return arr, is_looping


def process_route(args):
    in_arr, new_obstacle, start_pos = args
    new_arr = deepcopy(in_arr)
    new_arr[*new_obstacle] = 3
    _, is_loop = compute_route(new_arr, start_pos, False)
    return 1 if is_loop else 0


if __name__ == '__main__':
    with open('data/2024-06.txt', 'r') as f:
        data = f.readlines()

    char_map = {
        '.': 0,
        'X': 1,
        '^': 2,
        '#': 3,
    }
    all_rows = []
    for row in data:
        all_rows.append([char_map[char] for char in row.strip()])

    # part 1
    arr = np.array(all_rows)
    start_pos = np.array([np.where(arr == 2)[0][0].item(), np.where(arr == 2)[1][0].item()])
    arr_1, _ = compute_route(arr, start_pos, False)
    print('Part 1:', (arr_1 == 1).sum())

    # try to put an obstacle in any of the spots he took,
    # excluding his starting position
    arr_1[*start_pos] = 2
    obstacle_options = np.where(arr_1 == 1)
    obstacle_options = list(zip(*obstacle_options))

    start_time = time()
    count = 0
    for obstacle_placement in obstacle_options:
        new_arr = deepcopy(arr)
        new_arr[*obstacle_placement] = 3
        _, is_loop = compute_route(new_arr, start_pos, False)
        count += is_loop

    print('Part 2a')
    print('Result:', count)
    print('Time:', time() - start_time)

    # part 2 (fast?)
    start_time = time()
    with ProcessPoolExecutor() as executor:
        results = list(executor.map(
            process_route, [(arr, obstacle_placement, start_pos) for obstacle_placement in obstacle_options]))

    print('Part 2b')
    print('Result:', sum(results))
    print('Time:', time() - start_time)
