import heapq


def a_star(graph, heuristic, start, goal):
    open_set = []
    heapq.heappush(open_set, (heuristic[start], 0, start))  # (f_score, g_score, city)

    parent = {start: None}
    g_score = {start: 0}

    while open_set:
        f, current_g, current = heapq.heappop(open_set)

        if current == goal:
            # Reconstruct path
            path = []
            while current:
                path.append(current)
                current = parent[current]
            return path[::-1]

        for neighbor, cost in graph.get(current, []):
            tentative_g = current_g + cost
            if neighbor not in g_score or tentative_g < g_score[neighbor]:
                g_score[neighbor] = tentative_g
                f_score = tentative_g + heuristic.get(neighbor, float('inf'))
                heapq.heappush(open_set, (f_score, tentative_g, neighbor))
                parent[neighbor] = current

    return None  # No path found

city_graph = {
    "A": [("B", 1), ("C", 7)],
    "B": [("A", 1), ("D", 2), ("E", 5)],
    "C": [("A", 7), ("F", 3)],
    "D": [("B", 2)],
    "E": [("B", 5), ("F", 1)],
    "F": [("C", 3), ("E", 1), ("G", 2)],
    "G": [("F", 2)]
}

heuristic = {
    "A": 7,
    "B": 6,
    "C": 4,
    "D": 3,
    "E": 2,
    "F": 1,
    "G": 0  # Goal node has 0 heuristic
}

start_city = "A"
goal_city = "G"

path = a_star(city_graph, heuristic, start_city, goal_city)
if path:
    print(f"A* path from {start_city} to {goal_city}: {' -> '.join(path)}")
else:
    print("No path found.")