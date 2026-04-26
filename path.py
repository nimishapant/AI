from collections import deque

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

def bfs(start, goal):
    visited = set()  # Set to track visited states
    parent = {}  # To track the parent of each state
    queue = deque([(start, [])])  # Queue holds tuples of (state, path_so_far)

    visited.add(start)

    while queue:
        current, path = queue.popleft()

        if current == goal:
            return path  # Goal found, return the path

        # Generate next states
        index = current.index(0)  # Find the blank tile's index
        for move, swap_with in MOVE_MAP[index].items():
            new_state = list(current)
            new_state[index], new_state[swap_with] = new_state[swap_with], new_state[index]
            new_state = tuple(new_state)  # Convert back to tuple for immutability

            if new_state not in visited:
                visited.add(new_state)
                parent[new_state] = (current, move)  # Track parent and move
                queue.append((new_state, path + [move]))

    return None  # No solution if we exhaust the queue

def reconstruct_path(parent, goal, start):
    path = []
    current = goal
    while current != start:
        prev, move = parent[current]
        path.append(move)
        current = prev
    path.reverse()
    return path

start_state = (1, 2, 3,
               4, 0, 5,
               6, 7, 8)  # Example start state

goal_state = (1, 2, 3,
              4, 5, 6,
              7,8,0)  # Goal state

# Run BFS
bfs_path = bfs(start_state, goal_state)
if bfs_path:
    print("BFS Path:", bfs_path)
else:
    print("No solution found using BFS.")