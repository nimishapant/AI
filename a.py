
# Name: Nimisha Pant

import heapq

# Heuristic function (Hamming Distance)
def heuristic(state, goal):
    return sum(1 for i in range(len(state)) if state[i] != goal[i])

def astar(start, goal):
    pq = []
    heapq.heappush(pq, (0, start, []))
    visited = set()

    while pq:
        cost, state, path = heapq.heappop(pq)

        if state in visited:
            continue

        visited.add(state)

        # Goal check
        if state == goal:
            return path + [state]

        # Generate neighbors by flipping each bit
        for i in range(len(state)):
            new_state = list(state)
            new_state[i] = '1' if state[i] == '0' else '0'
            new_state = ''.join(new_state)

            g = len(path) + 1
            h = heuristic(new_state, goal)

            heapq.heappush(pq, (g + h, new_state, path + [state]))

    return None


# Initial and Goal states
start = "10110101"
goal = "01100011"

result = astar(start, goal)

if result:
    print("Solution path:")
    for step in result:
        print(step)
else:
    print("No solution found")

