import heapq

# Move map: index → possible moves (index changes)
MOVE_MAP = {
    0: {'right': 1, 'down': 3},
    1: {'left': 0, 'right': 2, 'down': 4},
    2: {'left': 1, 'down': 5},
    3: {'up': 0, 'right': 4, 'down': 6},
    4: {'up': 1, 'left': 3, 'right': 5, 'down': 7},
    5: {'up': 2, 'left': 4, 'down': 8},
    6: {'up': 3, 'right': 7},
    7: {'up': 4, 'left': 6, 'right': 8},
    8: {'up': 5, 'left': 7}
}



def manhattan_distance(state, goal):
    goal_positions = {value: (idx // 3, idx % 3) for idx, value in enumerate(goal) if value != 0}
    distance = 0
    for idx, value in enumerate(state):
        if value == 0:
            continue
        current_pos = (idx // 3, idx % 3)
        goal_pos = goal_positions[value]
        distance += abs(current_pos[0] - goal_pos[0]) + abs(current_pos[1] - goal_pos[1])
    return distance


def a_star(start, goal):
    open_set = []
    heapq.heappush(open_set, (manhattan_distance(start, goal), 0, start, []))
    visited = set()

    while open_set:
        f, g, current, path = heapq.heappop(open_set)

        if current == goal:
            return path

        if current in visited:
            continue
        visited.add(current)

        index = current.index(0)
        for move, swap_with in MOVE_MAP[index].items():
            new_state = list(current)
            new_state[index], new_state[swap_with] = new_state[swap_with], new_state[index]
            new_state_tuple = tuple(new_state)

            if new_state_tuple not in visited:
                new_path = path + [move]
                h = manhattan_distance(new_state_tuple, goal)
                heapq.heappush(open_set, (g + 1 + h, g + 1, new_state_tuple, new_path))

    return None  # No solution

# Example usage:
start_state = (1, 2, 3,
               4, 0, 5,
               6, 7, 8)

goal_state = (1, 3, 4,
              2, 6, 7,
              5, 8, 0)

solution_path = a_star(start_state, goal_state)
if solution_path:
    print("A* Solution (moves):", solution_path)
    print("Steps:", len(solution_path))
else:
    print("No solution found.")